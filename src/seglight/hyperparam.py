import json
import os
from collections.abc import Callable

import lightning as L
import optuna
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.core.module import LightningModule
from optuna import Study
from optuna.integration import PyTorchLightningPruningCallback
from tqdm import tqdm

from seglight.data import TrainTestDataModule


class OptunaLightningTuner:
    def __init__(
        self,
        model_builder: Callable,
        model_class: type[LightningModule],
        loss_fn,
        datamodule: TrainTestDataModule,
        param_search_space: dict,  # dict: {param_name: list_of_values}
        direction: str = "minimize",  # "minimize" or "maximize"
        max_epochs: int = 100,
        accelerator: str = "cpu",  # "cpu", "cuda"
        devices: int = 1,
        monitor_metric: str = "val_loss",
        eval_metrics: Callable | dict[str, Callable] | None = None,
        callbacks: (
            list | None
        ) = None,  # Is using optuna do nto need to define a ModelCheckpoint callback
        check_val_every_n_epoch: int = 1,
        log_every_n_steps: int = 1,  # log after n steps (batches)
        model_dir: str = "model_checkpoint",  # directory to save models
        study_name: str = "seglight_tuning",
    ):
        self.model_builder = model_builder
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.datamodule = datamodule
        self.param_search_space = param_search_space
        self.direction = direction
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.monitor_metric = monitor_metric
        self.eval_metrics = eval_metrics
        self.callbacks = callbacks if callbacks is not None else []
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.log_every_n_steps = log_every_n_steps
        self.model_dir = model_dir
        self.study_name = study_name

    def _prepare_trial_components(
        self, trial: optuna.trial.Trial
    ) -> tuple[LightningModule, list[Callback]]:
        params = {
            param: trial.suggest_categorical(param, choices)
            for param, choices in self.param_search_space.items()
        }

        model = self.model_builder(params, self.loss_fn)

        filename = f"trial_{trial.number}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename=filename,
            save_top_k=1,
            monitor=self.monitor_metric,
        )

        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor=self.monitor_metric
        )

        trainer_callbacks = [
            *self.callbacks,
            pruning_callback,
            checkpoint_callback,
        ]

        return model, trainer_callbacks

    def _save_to_pt(self, trial: optuna.trial.Trial, checkpoint_callback, model):
        best_model_path = checkpoint_callback.best_model_path
        best_model_state = torch.load(best_model_path)["state_dict"]

        model.load_state_dict(best_model_state)
        model.eval()

        filename = f"model_trial_{trial.number}.pt"
        model_path = os.path.join(self.model_dir, filename)

        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, model_path)

        print(f"Saved scripted model to {model_path}")

    def _evaluate_metric(self, checkpoint_callback, model):
        best_model_path = checkpoint_callback.best_model_path
        best_model_state = torch.load(best_model_path)["state_dict"]

        model.load_state_dict(best_model_state)
        model.eval()
        model.to(self.accelerator)

        self.datamodule.setup("predict")
        test_loader = self.datamodule.predict_dataloader()

        iou_metric = self.eval_metrics.to(self.accelerator)
        iou_metric.reset()

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(self.accelerator)
                targets = targets.to(self.accelerator)

                preds = model(inputs)
                iou_metric.update(preds, targets)

        return iou_metric.compute().item()

    def _save_to_json(self, study: Study):
        filename = f"{self.study_name}_results.json"
        used_metric = (
            self.monitor_metric
            if self.eval_metrics is None
            else "eval_metric (e.g. IoU)"
        )

        results = {
            "evaluation_metric_used": used_metric,
            "best_trial": study.best_trial.params,
            "trials": [
                {
                    "number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": str(trial.state),
                }
                for trial in study.trials
            ],
        }

        json_path = os.path.join(self.model_dir, filename)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Study results saved to {filename}")

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna hyperparameter tuning.

        Args:
            trial (optuna.trial.Trial): A single trial instance.

        Returns:
            float: The value of the metric used for optimization.
        """

        model, trainer_callbacks = self._prepare_trial_components(trial)

        trainer = L.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            max_epochs=self.max_epochs,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            log_every_n_steps=self.log_every_n_steps,
            callbacks=trainer_callbacks,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, datamodule=self.datamodule)

        metric_value = trainer.callback_metrics.get(self.monitor_metric)
        if metric_value is None:
            raise ValueError(
                f"Metric '{self.monitor_metric}' not found in callback_metrics."
            )
        checkpoint_callback = next(
            cb for cb in trainer_callbacks if isinstance(cb, ModelCheckpoint)
        )

        self._save_to_pt(trial, checkpoint_callback, model)
        if self.eval_metrics is None:
            value_for_optuna = metric_value.item()
        else:
            value_for_optuna = self._evaluate_metric(checkpoint_callback, model)

        return value_for_optuna

    def run_study(
        self,
        n_trials: int = 10,
        study_name: str | None = None,
        storage: str = "sqlite:///optuna_study.db",
        sampler: str = "grid",
        timeout: int | None = None,
    ) -> Study:
        """
        Runs an Optuna study to optimize hyperparameters for the model.

        Args:
            n_trials: Number of trials to run.
            study_name: Name of the Optuna study.
            storage: URL for persistent storage.
            sampler: One of ['grid', 'random', 'tpe'].
            timeout: Maximum time in seconds for the study.
        """
        if study_name is None:
            study_name = self.study_name
        if sampler == "grid":
            sampler_obj = optuna.samplers.GridSampler(self.param_search_space)
        elif sampler == "random":
            sampler_obj = optuna.samplers.RandomSampler()
        elif sampler == "tpe":
            sampler_obj = optuna.samplers.TPESampler()
        else:
            raise ValueError(f"Invalid sampler: {sampler}")

        study = optuna.create_study(
            sampler=sampler_obj,
            study_name=study_name,
            storage=storage,
            direction=self.direction,
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        self._save_to_json(study)

        return study

    def predict(self, datamodule, study):
        trial_num = study.best_trial.number
        path_of_model = os.path.join(self.model_dir, f"model_trial_{trial_num}.pt")
        model = torch.jit.load(path_of_model)
        model.eval()

        datamodule.setup("predict")
        loader = datamodule.predict_dataloader()

        preds = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, tuple | list) else batch
                out = model(x)
                preds.append(out.cpu().numpy())

        return [p for b in preds for p in b]

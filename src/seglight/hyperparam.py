import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import lightning as L
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.core.module import LightningModule
from optuna import Study
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import BaseSampler

from seglight.data import TrainTestDataModule


def setup_logging():
    level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


logger = logging.getLogger(__name__)


@dataclass
class TunerConfig:
    direction: str = "minimize"  # "minimize" or "maximize"
    max_epochs: int = 100
    accelerator: str = "cpu"  # "cpu", "cuda",...
    devices: int = 1
    monitor_metric: str = "val_loss"
    eval_metrics: Callable | None = None
    callbacks: list[Callback] | None = (
        None  # dont need to define a ModelCheckpoint callback
    )
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 1  # log after n steps (batches)
    num_sanity_val_steps: int = 0  # number of validation steps to run before training
    ckpt_dir: str = "checkpoints"
    model_dir: str = "output"
    study_name: str = "seglight_tuning"
    save_top_k: int = 1


class OptunaLightningTuner:
    def __init__(
        self,
        model_builder: Callable,
        model_class: type[LightningModule],
        loss_fn: Callable,
        datamodule: TrainTestDataModule,
        param_search_space: dict,  # dict: {param_name: list_of_values}
        config: TunerConfig,
    ):
        self.model_builder = model_builder
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.datamodule = datamodule
        self.param_search_space = param_search_space
        self.config = config

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
            dirpath=self.config.ckpt_dir,
            filename=filename,
            save_top_k=self.config.save_top_k,
            monitor=self.config.monitor_metric,
        )

        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor=self.config.monitor_metric
        )

        trainer_callbacks = [
            *(self.config.callbacks or []),
            pruning_callback,
            checkpoint_callback,
        ]

        return model, trainer_callbacks

    def _load_best_model(self, checkpoint_callback, model):
        best_model_path = checkpoint_callback.best_model_path
        best_model_state = torch.load(best_model_path)["state_dict"]
        model.load_state_dict(best_model_state)
        model.eval()
        return model

    def _save_to_pt(self, trial: optuna.trial.Trial, model):
        filename = f"model_trial_{trial.number}.pt"
        model_path = os.path.join(self.config.model_dir, filename)
        os.makedirs(self.config.model_dir, exist_ok=True)
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, model_path)

        logger.info(f"Saved model to {model_path}")

    def _evaluate_metric(self, model) -> float:
        model.to(self.config.accelerator)

        self.datamodule.setup("predict")
        test_loader = self.datamodule.predict_dataloader()

        metric = self.config.eval_metrics.to(self.config.accelerator)
        metric.reset()

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(self.config.accelerator)
                targets = targets.to(self.config.accelerator)
                preds = model(inputs)
                metric.update(preds, targets)

        return metric.compute().item()

    def _save_to_json(self, study: Study):
        filename = f"{self.config.study_name}_results.json"
        used_metric = (
            self.config.monitor_metric
            if self.config.eval_metrics is None
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

        json_path = os.path.join(self.config.model_dir, filename)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Study results saved to {json_path}")

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
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            max_epochs=self.config.max_epochs,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            log_every_n_steps=self.config.log_every_n_steps,
            callbacks=trainer_callbacks,
            num_sanity_val_steps=self.config.num_sanity_val_steps,
        )

        trainer.fit(model, datamodule=self.datamodule)

        metric_value = trainer.callback_metrics.get(self.config.monitor_metric)
        if metric_value is None:
            raise ValueError(
                f"Metric '{self.config.monitor_metric}' not found in callback_metrics."
            )
        checkpoint_callback = next(
            cb for cb in trainer_callbacks if isinstance(cb, ModelCheckpoint)
        )
        best_model = self._load_best_model(checkpoint_callback, model)

        self._save_to_pt(trial, best_model)
        if self.config.eval_metrics is None:
            value_for_optuna = metric_value.item()
        else:
            value_for_optuna = self._evaluate_metric(best_model)

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
        sampler_obj: BaseSampler
        if study_name is None:
            study_name = self.config.study_name
        if sampler == "grid":
            sampler_obj = optuna.samplers.GridSampler(self.param_search_space)
        elif sampler == "random":
            sampler_obj = optuna.samplers.RandomSampler()
        elif sampler == "tpe":
            sampler_obj = optuna.samplers.TPESampler()
        else:
            raise ValueError(f"Invalid sampler: {sampler}")

        logger.info(f"Starting study '{study_name}' with sampler: {sampler}")

        study = optuna.create_study(
            sampler=sampler_obj,
            study_name=study_name,
            storage=storage,
            direction=self.config.direction,
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        self._save_to_json(study)

        return study

    def predict(self, datamodule, study) -> list[np.ndarray]:
        trial_num = study.best_trial.number
        path_of_model = os.path.join(
            self.config.model_dir, f"model_trial_{trial_num}.pt"
        )
        model = torch.jit.load(path_of_model)
        model.eval()
        logger.info(f"Loaded model from {path_of_model} for prediction")

        datamodule.setup("predict")
        loader = datamodule.predict_dataloader()

        preds = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, tuple | list) else batch
                out = model(x)
                preds.append(out.cpu().numpy())
        logger.info("Prediction completed")
        return [p for b in preds for p in b]

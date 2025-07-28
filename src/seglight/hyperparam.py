import os
from collections.abc import Callable

import lightning as L
import optuna
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.core.module import LightningModule
from optuna import Study
from optuna.integration import PyTorchLightningPruningCallback

from seglight.data import TrainTestDataModule


class OptunaLightningTuner:
    def __init__(
        self,
        model_builder: Callable,
        loss_fn,
        datamodule: TrainTestDataModule,
        param_search_space: dict,  # dict: {param_name: list_of_values}
        direction: str = "minimize",  # "minimize" or "maximize"
        max_epochs: int = 100,
        accelerator: str = "cpu",  # "cpu", "gpu"
        devices: int = 1,
        monitor_metric: str = "val_loss",
        eval_metrics: Callable | dict[str, Callable] | None = None,
        callbacks: (
            list | None
        ) = None,  # Is using optuna do nto need to define a ModelCheckpoint callback
        check_val_every_n_epoch: int = 1,
        log_every_n_steps: int = 1,  # log after n steps (batches)
        model_dir: str = "model_checkpoint",  # directory to save models
    ):
        self.model_builder = model_builder
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

    def _prepare_trial_components(
        self, trial: optuna.trial.Trial
    ) -> tuple[LightningModule, list[Callback]]:
        """
        Prepare model and callbacks for a given Optuna trial.

        Args:
            trial (optuna.trial.Trial): The current Optuna trial.

        Returns:
            Tuple[LightningModule, List[Callback]]: The model instance and a
            list of callbacks.
        """

        # Suggest params
        params = {}
        # TODO - handle float or int params not just categorical
        for param, choices in self.param_search_space.items():
            params[param] = trial.suggest_categorical(param, choices)

        # Build model with suggested params
        model = self.model_builder(params, self.loss_fn)

        filename = f"trial_{trial.number}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename=filename,
            save_top_k=1,
            monitor=self.monitor_metric,
        )
        # to enable pruning and checking the intermediate results (val loss plots)
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor=self.monitor_metric
        )

        trainer_callbacks = [
            *self.callbacks,
            pruning_callback,
            checkpoint_callback,
        ]

        return model, trainer_callbacks

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna hyperparameter tuning.

        Args:
            trial (optuna.trial.Trial): A single trial instance.

        Returns:
            float: The value of the monitored metric used for optimization.
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
        best_model_path = checkpoint_callback.best_model_path

        # Load the best checkpoint instead of the last model state
        best_model_state = torch.load(best_model_path)["state_dict"]

        # Save it to a controlled location (optional)
        save_path = f"model_trial_{trial.number}.pt"
        torch.save(best_model_state, save_path)
        print(f"Saved best model state_dict from checkpoint to {save_path}")

        # TODO - here do also otehr metrics  after evaluation
        # fisrt put model into eval and then call metric whic is callable
        # so can be one Metric or dict of metrics and for eh it will return  some
        # float value
        # if None return val_loss
        # callable function can be tehre postprocessinf
        # and save and make soem good metadata into folder

        return metric_value.item()

    def run_study(
        self,
        n_trials: int = 10,
        study_name: str = "seglight_tuning",
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

        return study

    def get_best_model_path(self, study: optuna.Study) -> str:
        """Returns the path to the best model based on the study results."""
        best_trial = study.best_trial
        filename = f"trial_{best_trial.number}.ckpt"
        return os.path.join(self.model_dir, filename)

    def predict_best_model(self, study: optuna.Study) -> list:
        """
        Loads the best model from checkpoint and runs prediction on the datamodule.
        Returns:
            List of numpy arrays containing predictions.
        """
        best_params = study.best_trial.params
        model_path = self.get_best_model_path(study)

        # Get model class
        dummy_model = self.model_builder(best_params, self.loss_fn)
        model_class = type(dummy_model)

        # Load model from checkpoint
        try:
            model = model_class.load_from_checkpoint(model_path, loss_fn=self.loss_fn)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model checkpoint from {model_path}: {e}"
            ) from e

        model.eval()
        trainer = L.Trainer(accelerator=self.accelerator, devices=self.devices)
        with torch.no_grad():
            preds_tensors = trainer.predict(model, datamodule=self.datamodule)

        # Flatten predictions
        return [p for batch in preds_tensors for p in batch.cpu().numpy()]

from typing import Any, Dict
import logging

from allennlp.common.params import Params
from allennlp.data import DatasetReader, DataLoader
from allennlp.training.util import evaluate
from allennlp.training.metrics import Metric
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.callbacks.callback import TrainerCallback

logger = logging.getLogger(__name__)


@TrainerCallback.register("distributed_test")
class DistributedTestCallback(TrainerCallback):
    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        """
        This callback hook is called before the training is started.
        """
        self.trainer = trainer
        if is_primary:
            params = Params.from_file(self.serialization_dir + "/config.json")
            if params['test_data_path'] is None:
                self.test_data_loader = None
            else:
                dataset_reader = DatasetReader.from_params(params['dataset_reader'])
                self.test_data_loader = DataLoader.from_params(
                    params['data_loader'], reader=dataset_reader,
                    data_path=params['test_data_path']
                )
                self.test_data_loader.index_with(trainer.model.vocab)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each epoch.
        """
        if is_primary and self.test_data_loader:
            self.set_metric_status(trainer, True)
            evaluate(
                trainer.model, self.test_data_loader, trainer.cuda_device,
                output_file=f"{self.serialization_dir}/metrics_test_{epoch}.json"
            )
            self.set_metric_status(trainer, False)

    def on_end(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the final training epoch.
        """
        if is_primary:
            self.set_metric_status(trainer, True)

    def set_metric_status(self, trainer, status: bool):
        for k, v in trainer.model.__dict__.items():
            if isinstance(v, Metric) and "training_finished" in v.__dict__:
                logger.info("set `%s.training_finished = %s`", k, status)
                v.training_finished = status

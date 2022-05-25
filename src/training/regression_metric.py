from typing import Dict
from itertools import chain

import sklearn.metrics as skm

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics import Metric


@Metric.register("regression")
class RegressionMetric(Metric):

    def __init__(self) -> None:
        self._predictions = list()
        self._gold_labels = list()
        self.training_finished = False

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor):
        predictions, gold_labels = list(self.detach_tensors(predictions, gold_labels))
        self._gold_labels.extend(gold_labels.cpu().split(1))
        self._predictions.extend(predictions.cpu().split(1))

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if reset and is_distributed() and not self.training_finished:
            # gather counters only when reseting can speed up training.
            def _gather_list(obj):
                obj_list = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(obj_list, obj)
                return list(chain(*obj_list))

            self._predictions = _gather_list(self._predictions)
            self._gold_labels = _gather_list(self._gold_labels)

        predictions = torch.tensor(self._predictions)
        gold_labels = torch.tensor(self._gold_labels)
        ae = torch.abs(gold_labels - predictions)
        se = torch.pow(ae, 2)
        mse = torch.mean(se)
        metrics = {"mean_absolute_error": float(ae.mean())}
        metrics["mean_squared_error"] = float(mse)
        metrics["root_mean_squared_error"] = float(mse.sqrt())
        metrics["max_error"] = float(ae.max())
        metrics["r2_score"] = float(
            1 - se.sum() / torch.pow(gold_labels - predictions.mean(), 2).sum()
        )
        metrics["sklearn_r2_score"] = skm.r2_score(gold_labels, predictions)

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._predictions = list()
        self._gold_labels = list()

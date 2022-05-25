from typing import Dict, List, Optional, Set, Iterable
import logging
import os
import glob

from PIL import Image

import torch

from transformers import AutoFeatureExtractor

from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
from allennlp.data.fields import TensorField, MetadataField
from allennlp.data import Instance, Field

logger = logging.getLogger(__name__)


_valid_worker_map = {
    w: i for i, w in enumerate([
        1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 58, 59, 60,
        61, 63, 67
    ], 1)  # leave 0 as expert
}


@DatasetReader.register("label_me_crowd")
class LabelMeCrowdDatasetReader(DatasetReader):
    def __init__(
        self,
        model_name: str,
        exclude: Optional[str] = 'none',
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._featurizer = AutoFeatureExtractor.from_pretrained(model_name)
        self._label_namespace = label_namespace
        self.exclude_workers: Set[int] = {}  # self.exclude_ID[exclude]

    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:
        if "labels" in file_path:
            names_path = file_path.replace("labels", "filenames")
        else:
            names_path = os.path.dirname(file_path) + "/filenames_train.txt"

        if "valid" in file_path:
            part = "valid"
        elif "test" in file_path:
            part = "test"
        else:
            part = "train"
        name_to_path = {os.path.basename(p): p for p in glob.glob(
            f"{os.path.dirname(file_path)}/{part}/*/*.jpg"
        )}

        with open(names_path, 'r') as names_file, open(file_path, 'r') as data_file:
            logger.info("Reading instances from lines in: %s, %s", file_path, names_path)
            for name, line, in zip(names_file, data_file):
                name, labels = name.strip(), line.split()
                image = Image.open(name_to_path[name])
                feature = self._featurizer(image, return_tensors="pt").data
                if "answers" in file_path and "mv.txt" not in file_path:
                    for ins in self.text_to_instances(feature, labels, name):
                        yield ins
                else:
                    yield self.text_to_instance(feature, labels[0], name)

    def text_to_instances(
        self,
        feature: Dict[str, torch.Tensor],
        labels: List[str],
        filename: str
    ) -> Iterable[Instance]:
        """
        we leave worker id 0 as the expert.
        """
        for i, label in enumerate(labels):
            if label == '-1':
                continue
            worker = _valid_worker_map[i]
            if worker in self.exclude_workers:
                continue
            yield self.text_to_instance(feature, label, filename, worker)

    def text_to_instance(
        self,
        feature: Dict[str, torch.Tensor],
        label: str,
        filename: str,
        worker: int = -1
    ) -> Instance:
        """
        worker == -1 means we don't use annotator information.
        """
        fields: Dict[str, Field] = {
            k: TensorField(v.squeeze_()) for k, v in feature.items()
        }
        fields["label"] = TensorField(torch.tensor(int(label)), dtype=torch.long)
        fields["metadata"] = MetadataField({"filename": filename})
        fields["worker"] = TensorField(torch.tensor(worker), dtype=torch.long)
        return Instance(fields)

from typing import Dict, List, Optional, Set, Iterable
import logging
import os

import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
from allennlp.data.fields import TextField, TensorField, Field, MetadataField
from allennlp.data import Instance, TokenIndexer, Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("movie_reivew_crowd")
class MovieReivewCrowdDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        exclude: Optional[str] = 'none',
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.exclude_workers: Set[int] = {}  # self.exclude_ID[exclude]

    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:
        text_path = os.path.dirname(file_path) + "/texts_"
        text_path += ("test" if "test" in file_path else "train") + ".txt"
        with open(text_path, 'r') as text_file, open(file_path, 'r') as data_file:
            logger.info("Reading instances from lines in: %s, %s", file_path, text_path)
            for text_line, data_line, in zip(text_file, data_file):
                text, labels = text_line.strip(), data_line.split()
                if "answers" in file_path:
                    for ins in self.text_to_instances(text, labels):
                        yield ins
                else:
                    yield self.text_to_instance(text, labels[0])

    def text_to_instances(self, text: str, labels: List[str]) -> Iterable[Instance]:
        """
        we leave worker id 0 as the expert.
        """
        for i, label in enumerate(labels):
            worker = i + 1
            if worker in self.exclude_workers:
                continue
            if label != '-1':
                yield self.text_to_instance(text, label, worker)

    def text_to_instance(self, text: str, label: str, worker: int = -1) -> Instance:
        """
        worker == -1 means we don't use annotator information.
        """
        tokens = self._tokenizer.tokenize(text)
        sequence = TextField(tokens, self._token_indexers)
        fields: Dict[str, Field] = {"tokens": sequence}
        fields["label"] = TensorField(torch.tensor(float(label)))
        fields["metadata"] = MetadataField({"text": text})
        fields["worker"] = TensorField(torch.tensor(worker), dtype=torch.long)
        return Instance(fields)

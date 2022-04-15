from typing import Dict, List, Optional, Set, Iterable, Tuple
import itertools
import logging

import torch
# from transformers import PreTrainedTokenizerFast
# from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
from allennlp.data.fields import TextField, SequenceLabelField, TensorField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


@DatasetReader.register("conll2003_ner_crowd")
class Conll2003NerCrowdDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    ```
    WORD POS-TAG CHUNK-TAG NER-TAG
    ```

    Registered as a `DatasetReader` with name "conll2003_ner".
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, required
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    label_namespace : `str`, optional (default=`labels`)
        Specifies the namespace for the chosen `tag_label`.
    """

    exclude_ID = {
        'none': set(),
        'bad': {24, 20, 42, 18, 11, 13, 46, 38, 35, 12, 37, 5, 16, 21, 14, 33},
        'small': set(i for i in range(48) if i not in {
            34, 2, 10, 8, 25, 23, 28, 20, 3, 44, 20, 12, 14}),
        'middle': set(i for i in range(48) if i not in {
            34, 2, 36, 8, 25, 23, 28, 44, 45, 20, 13, 21, 14}),
    }

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        exclude: Optional[str] = 'none',
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._token_indexers = token_indexers
        self.label_namespace = label_namespace
        self.exclude_workers: Set[int] = self.exclude_ID[exclude]

    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group lines into sentence chunks based on the divider.
            line_chunks = (
                lines
                for is_divider, lines in itertools.groupby(data_file, _is_divider)
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider
            )
            for lines in self.shard_iterable(line_chunks):
                fields = [line.strip().split() for line in lines]
                # unzipping trick returns tuples, but our Fields need lists
                tokens, *tags = [list(field) for field in zip(*fields)]
                if "answers" in file_path:
                    for ins in self.text_to_instances(tokens, tags):
                        yield ins
                else:
                    yield self.text_to_instance(tokens, tags[0])

    def text_to_instances(self, tokens, tag_mat) -> Iterable[Instance]:
        """
        we leave worker id 0 as the expert.
        """
        for i in range(len(tag_mat)):
            worker = i + 1
            if worker in self.exclude_workers:
                continue
            if len(set(tag_mat[i])) > 1:
                yield self.text_to_instance(tokens, tag_mat[i], worker)

    def text_to_instance(  # type: ignore
        self,
        words: List[str],
        tags: List[str],
        worker: int = -1
    ) -> Instance:
        """
        worker == -1 means we don't use annotator information.
        """
        # words, tags = self.expand_word_piece(words, tags)
        sequence = TextField([Token(w) for w in words], self._token_indexers)
        fields: Dict[str, Field] = {"tokens": sequence}
        fields["metadata"] = MetadataField({"words": words})
        fields["tags"] = SequenceLabelField(tags, sequence, self.label_namespace)
        fields["worker"] = TensorField(torch.tensor(worker), dtype=torch.long)
        return Instance(fields)

    # def expand_word_piece(self, words, tags) -> Tuple[List[str], List[str]]:
    #     tokenizer: PreTrainedTokenizerFast = self._token_indexers['bert']._tokenizer
    #     expanded_words, expanded_tags = list(), list()
    #     for word, tag in zip(words, tags):
    #         pieces = tokenizer.tokenize(word)
    #         if tag.startswith("B-"):
    #             i_tag = tag.replace("B-", "I-")
    #             piece_tags = [tag] + [i_tag for _ in range(len(pieces) - 1)]
    #         else:
    #             piece_tags = [tag for _ in pieces]
    #         expanded_words.extend(pieces)
    #         expanded_tags.extend(piece_tags)

    #     return expanded_words, expanded_tags

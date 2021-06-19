"""
PGN adapter-BERT
"""

from typing import Dict, Optional, Any, Set

from overrides import overrides
import torch
from torch.nn import Parameter

# from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model

from allennlpadd.modules.token_embedders import PgnAdapterTransformerEmbedder
from .crf_tagger import CrfTagger


@Model.register("pgn_crf_tagger")
class PgnCrfTagger(CrfTagger):
    """
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Dict[str, Any],
        worker_num: int,
        supervised: bool = False,
        crowd_test: bool = False,
        exclude_workers: Set[int] = (0,),
        worker: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, text_field_embedder, encoder, **kwargs)

        self.supervised = supervised
        self.crowd_test = crowd_test
        self.worker = worker

        if not isinstance(exclude_workers, set):
            exclude_workers = set(exclude_workers)
        if not supervised:
            # we leave id=0 as the expert.
            mean_ids = set(i for i in range(1, worker_num)) - set(exclude_workers)
            self.mean_ids = Parameter(
                torch.tensor(list(mean_ids), dtype=torch.int), requires_grad=False)

    # @property won't work, see https://github.com/pytorch/pytorch/issues/50292
    def pgn_adapter_bert(self) -> PgnAdapterTransformerEmbedder:
        return self.text_field_embedder._token_embedders['bert'].matched_embedder

    @overrides
    def forward(self, worker: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        if self.training or self.crowd_test:
            self.pgn_adapter_bert().preset_domain = None
        else:
            domain_embedding = self.pgn_adapter_bert().domain_embedding
            if isinstance(self.worker, int):
                embedding = domain_embedding.weight[self.worker]
            elif self.supervised:
                embedding = domain_embedding.weight[0]
            else:
                embedding = domain_embedding(self.mean_ids).mean(0)
            self.pgn_adapter_bert().preset_domain = True
            self.pgn_adapter_bert().generate_parameters(embedding)

        return super().forward(domain=worker, **kwargs)

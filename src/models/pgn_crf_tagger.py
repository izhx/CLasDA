"""
PGN adapter-BERT
"""

from typing import Dict, Optional, Any, Set

import torch
import torch.nn as nn
from torch.nn import Parameter

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model

from ..modules.adapter_embedder import AdapterTransformerEmbedder
from .crf_tagger import CrfTagger


@Model.register("pgn_crf_tagger")
class PgnCrfTagger(CrfTagger):
    """
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        worker_num: int,
        worker_dim: int = 8,
        pgn_layers: int = 12,
        share_param: bool = False,
        supervised: bool = False,
        crowd_test: bool = False,
        exclude_workers: Set[int] = set(),
        worker: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, text_field_embedder, encoder, **kwargs)
        self.worker_num = worker_num
        self.pgn_layers = pgn_layers
        self.share_param = share_param
        self.supervised = supervised
        self.crowd_test = crowd_test
        self.worker = worker

        if not isinstance(exclude_workers, set):
            exclude_workers = set(exclude_workers)
        if not supervised:
            exclude_workers.add(0)  # we leave id=0 as the expert.
            # annotator IDs for estimating the expert
            workers = [i for i in range(0, worker_num) if i not in exclude_workers]
            self.workers = Parameter(
                torch.tensor(workers, dtype=torch.long), requires_grad=False
            )

        if pgn_layers > self.adapter_bert().adapter_layers:
            raise ConfigurationError(
                f"pgn_layers {pgn_layers} should <= adapter_layers "
                f"{self.adapter_bert().adapter_layers}"
            )

        self.worker_embedding = nn.Embedding(worker_num, worker_dim)  # max_norm=1.0

        hidden_size = self.adapter_bert().transformer_model.config.hidden_size
        adapter_size: int = self.adapter_bert().adapters[0][0].adapter_size
        size = [2] if share_param else [pgn_layers, 2]
        weights = dict(
            weight_down=Parameter(torch.Tensor(
                *size, adapter_size, hidden_size, worker_dim)),
            weight_up=Parameter(torch.Tensor(
                *size, hidden_size, adapter_size, worker_dim))
        )
        if self.adapter_bert().adapters[0][0].bias:
            weights.update(
                bias_down=Parameter(torch.zeros(
                    *size, adapter_size, worker_dim)),
                bias_up=Parameter(torch.zeros(
                    *size, hidden_size, worker_dim))
            )
        self.weights = nn.ParameterDict(weights)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weights.weight_down, std=1e-3)
        nn.init.normal_(self.weights.weight_up, std=1e-3)

    # @property won't work, see https://github.com/pytorch/pytorch/issues/50292
    def adapter_bert(self) -> AdapterTransformerEmbedder:
        return self.text_field_embedder._token_embedders['bert'].matched_embedder

    def forward(self, worker: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        if self.training or self.crowd_test:
            embedding = self.worker_embedding(worker)
        elif isinstance(self.worker, int):
            embedding = self.worker_embedding.weight[self.worker]
        elif self.supervised:
            embedding = self.worker_embedding.weight[0]
        else:
            embedding = self.worker_embedding(self.workers).mean(0)
        self.generate_parameters(embedding)
        return super().forward(**kwargs)

    def generate_parameters(self, embedding: torch.Tensor):
        def batch_matmul(w: torch.Tensor, e):
            ALPHA = "ijklmnopqrstuvwxyz"
            dims = ALPHA[:w.dim() - 1]
            i = 1 if self.share_param else 2
            return torch.einsum(f"{dims}a,ba->{dims[:i] + 'b' + dims[i:]}", w, e)

        matmul = batch_matmul if embedding.dim() == 2 else torch.matmul
        embedding = embedding.softmax(-1)
        weights = {k: matmul(v, embedding) for k, v in self.weights.items()}

        for i, adapters in enumerate(self.adapter_bert().adapters[-self.pgn_layers:]):
            for j, adapter in enumerate(adapters):
                for k, v in weights.items():
                    param = v[j] if self.share_param else v[i, j]
                    setattr(adapter, k, param)
        return

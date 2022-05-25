from typing import Any, Dict, Optional, Set

import torch
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    Backbone, TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, FeedForward
)
from .crf_tagger import CrfTagger
from .image_classifier import ImageClassifier
from .text_regressor import TextRegressor
from ..modules.adapter import AdapterMixin


class PgnMixin:
    def __init__(
        self,
        worker_num: int,
        worker_dim: int = 8,
        pgn_layers: int = 12,
        share_param: bool = False,
        supervised: bool = False,
        crowd_test: bool = False,
        exclude_workers: Set[int] = set(),
        worker: Optional[int] = None,
    ) -> None:
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
            self.workers = nn.Parameter(
                torch.tensor(workers, dtype=torch.long), requires_grad=False
            )

        if pgn_layers > self.backbone().adapter_layers():
            raise ValueError(
                f"pgn_layers {pgn_layers} should <= adapter_layers "
                f"{self.backbone().adapter_layers()}"
            )

        self.worker_embedding = nn.Embedding(worker_num, worker_dim)  # max_norm=1.0

        hidden_size = self.backbone().get_output_dim()
        adapter_size: int = self.backbone().adapter_size()
        size = [2] if share_param else [pgn_layers, 2]
        weights = dict(
            weight_down=nn.Parameter(
                torch.Tensor(*size, adapter_size, hidden_size, worker_dim)
            ),
            weight_up=nn.Parameter(
                torch.Tensor(*size, hidden_size, adapter_size, worker_dim)
            )
        )
        if self.backbone().adapters[0][0].bias:
            weights.update(
                bias_down=nn.Parameter(
                    torch.zeros(*size, adapter_size, worker_dim)
                ),
                bias_up=nn.Parameter(
                    torch.zeros(*size, hidden_size, worker_dim)
                )
            )
        self.weights = nn.ParameterDict(weights)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weights.weight_down, std=1e-3)
        nn.init.normal_(self.weights.weight_up, std=1e-3)

    def backbone(self) -> AdapterMixin:
        raise NotImplementedError

    def set_worker(self, worker: torch.LongTensor):
        if self.training or self.crowd_test:
            embedding = self.worker_embedding(worker)
        elif isinstance(self.worker, int):
            embedding = self.worker_embedding.weight[self.worker]
        elif self.supervised:
            embedding = self.worker_embedding.weight[0]
        else:
            embedding = self.worker_embedding(self.workers).mean(0)
        self.generate_parameters(embedding)

    def generate_parameters(self, embedding: torch.Tensor):
        def batch_matmul(w: torch.Tensor, e):
            ALPHA = "ijklmnopqrstuvwxyz"
            dims = ALPHA[:w.dim() - 1]
            i = 1 if self.share_param else 2
            return torch.einsum(f"{dims}a,ba->{dims[:i] + 'b' + dims[i:]}", w, e)

        matmul = batch_matmul if embedding.dim() == 2 else torch.matmul
        embedding = embedding.softmax(-1)
        weights = {k: matmul(v, embedding) for k, v in self.weights.items()}

        for i, adapters in enumerate(self.backbone().adapters[-self.pgn_layers:]):
            for j, adapter in enumerate(adapters):
                for k, v in weights.items():
                    param = v[j] if self.share_param else v[i, j]
                    setattr(adapter, k, param)
        return


@Model.register("pgn_crf_tagger")
class PgnCrfTagger(CrfTagger, PgnMixin):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        pgn: Dict[str, Any],
        embedder_key: str,
        **kwargs,
    ) -> None:
        self._embedder_key = embedder_key
        super().__init__(vocab, text_field_embedder, encoder, **kwargs)
        PgnMixin.__init__(self, **pgn)

    def backbone(self):
        return self.text_field_embedder._token_embedders[self._embedder_key].matched_embedder

    def forward(self, worker: torch.Tensor, **kwargs):
        self.set_worker(worker)
        return CrfTagger.forward(self, **kwargs)


@Model.register("pgn_image_classifier")
class PgnImageClassifier(ImageClassifier, PgnMixin):
    def __init__(
        self,
        vocab: Vocabulary,
        backbone: Backbone,
        pgn: Dict[str, Any],
        feedforward: Optional[FeedForward] = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, backbone, feedforward, **kwargs)
        PgnMixin.__init__(self, **pgn)

    def backbone(self):
        return self._backbone

    def forward(self, worker: torch.Tensor, **kwargs):
        self.set_worker(worker)
        return ImageClassifier.forward(self, **kwargs)


@Model.register("pgn_text_regressor")
class PgnTextRegressor(TextRegressor, PgnMixin):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        pgn: Dict[str, Any],
        embedder_key: str,
        feedforward: Optional[FeedForward] = None,
        **kwargs
    ) -> None:
        self._embedder_key = embedder_key
        super().__init__(vocab, text_field_embedder, seq2vec_encoder, feedforward, **kwargs)
        PgnMixin.__init__(self, **pgn)

    def backbone(self):
        return self._text_field_embedder._token_embedders[self._embedder_key]

    def forward(self, worker: torch.Tensor, **kwargs):
        self.set_worker(worker)
        return TextRegressor.forward(self, **kwargs)

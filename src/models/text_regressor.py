from typing import Dict, Optional

import torch
from torch import nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask

from ..training.regression_metric import RegressionMetric


@Model.register("text_regressor")
class TextRegressor(Model):
    """
    This `Model` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    a linear classification layer, which projects into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.

    Registered as a `Model` with name "basic_classifier".

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        scale: int = 10,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        self._dropout = nn.Dropout(dropout) if dropout else dropout
        self._regressor = nn.Linear(text_field_embedder.get_output_dim(), 1)
        self._metric = RegressionMetric()
        self._loss = nn.MSELoss()
        self.scale = scale

        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        label: torch.IntTensor = None,
        metadata: MetadataField = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `scores` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                the predictions.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        scores = self._regressor(embedded_text).squeeze(-1)
        output_dict = {"scores": scores}

        if label is not None:
            loss = self._loss(scores, label)
            output_dict["loss"] = loss
            self._metric(scores * self.scale, label * self.scale)

        if metadata is not None:
            output_dict["text"] = [x["text"] for x in metadata]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metric.get_metric(reset)

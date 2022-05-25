from typing import Dict, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Backbone
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy

ID_TO_LABEL = {
    0: 'highway', 1: 'insidecity', 2: 'tallbuilding', 3: 'street', 4: 'forest',
    5: 'coast',  6: 'mountain', 7: 'opencountry'
}


@Model.register("image_classifier")
class ImageClassifier(Model):
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
    backbone : `Backbone`
        Used to embed the input text into a `TextField`
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        backbone: Backbone,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._backbone = backbone
        self._feedforward = feedforward
        if feedforward is not None:
            classifier_input_dim = feedforward.get_output_dim()
        else:
            classifier_input_dim = backbone.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._classifier = torch.nn.Linear(classifier_input_dim, len(ID_TO_LABEL))
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(  # type: ignore
        self,
        label: torch.IntTensor = None,
        metadata: MetadataField = None,
        **kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        features = self._backbone(kwargs)

        if self._dropout:
            features = self._dropout(features)

        if self._feedforward is not None:
            features = self._feedforward(features)

        logits = self._classifier(features)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        if metadata is not None:
            output_dict["filename"] = [x["filename"] for x in metadata]

        return output_dict

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = ID_TO_LABEL[label_idx]
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics

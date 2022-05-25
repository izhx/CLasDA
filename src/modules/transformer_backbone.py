from typing import Dict, Optional, Any, List
import logging
import inspect

import torch
import transformers
from transformers import PreTrainedModel
from allennlp.modules.backbones.backbone import Backbone

logger = logging.getLogger(__name__)


@Backbone.register("transformer")
class TransformerBackbone(Backbone):
    def __init__(
        self,
        model_name: str,
        class_name: str = "AutoModel",
        output_name: str = "last_hidden_state",
        train_parameters: bool = True,
        eval_mode: bool = False,
        sub_module: str = None,
        sub_modules_to_drop: Optional[List[str]] = None,
        gradient_checkpointing: bool = False,
        transformer_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self._output_name = output_name

        model_class = getattr(transformers, class_name)
        logger.info("loading %s", model_class)
        self._transformer_model: PreTrainedModel = model_class.from_pretrained(
            model_name, **(transformer_kwargs or {}),
        )
        self.config = self._transformer_model.config

        if gradient_checkpointing:
            self._transformer_model.gradient_checkpointing_enable()

        if sub_module:
            assert hasattr(self._transformer_model, sub_module)
            self._transformer_model = getattr(self._transformer_model, sub_module)

        if sub_modules_to_drop:
            for sub in sub_modules_to_drop:
                setattr(self._transformer_model, sub, None)

        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self._transformer_model.parameters():
                param.requires_grad = False

        self.eval_mode = eval_mode
        if eval_mode:
            self._transformer_model.eval()

        self.params = inspect.signature(self._transformer_model.forward).parameters

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        input_dict = {k: v for k, v in inputs.items() if k in self.params}
        output = self._transformer_model(**input_dict)
        return getattr(output, self._output_name)

    def get_output_dim(self):
        # I'm not sure if this works for all models; open an issue on github if
        # you find a case where it doesn't work.
        return self.config.hidden_size

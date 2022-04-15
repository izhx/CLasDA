"""
Adapter transformers for AllenNLP.
Parameter-Efficient Transfer Learning for NLP. https://arxiv.org/abs/1902.00751
"""

from typing import Optional, Dict, Any, Union, List

import torch
import torch.nn as nn
from torch.nn.functional import linear

from transformers.models.bert.modeling_bert import BertModel, BertOutput, BertSelfOutput
from transformers.models.electra.modeling_electra import ElectraModel
from transformers.models.roberta.modeling_roberta import RobertaModel

from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import TokenEmbedder, PretrainedTransformerEmbedder
from allennlp.nn import Activation


@TokenEmbedder.register("adapter_transformer")
class AdapterTransformerEmbedder(PretrainedTransformerEmbedder):
    """
    目前只针对 *BERT 结构，插入adapter.
    """
    def __init__(
        self,
        model_name: str,
        *,
        adapter_layers: int = 12,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        external_param: Union[bool, List[bool]] = False,
        max_length: int = None,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name,
            max_length=max_length,
            train_parameters=False,
            last_layer_only=True,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )
        self.adapters = insert_adapters(
            adapter_layers, adapter_kwargs, external_param, self.transformer_model)
        self.adapter_layers = adapter_layers
        self.adapter_kwargs = adapter_kwargs


def insert_adapters(
    adapter_layers: int, adapter_kwargs: Optional[Dict[str, Any]],
    external_param: Union[bool, List[bool]], transformer_model: BertModel
) -> nn.ModuleList:
    """
    初始化 adapters, 插入到 BERT, 并返回 adapters. 目前只支持 *BERT 结构!

    # Parameters

    adapter_layers : `int`, required.
        从 BERT 最后一层开始，多少层插入adapter。
    adapter_kwargs : `Dict`, required.
        初始化 `Adapter` 的参数。
    external_param : `Union[bool, List[bool]]`
        adapter 的参数是否留空以便外部注入。
    transformer_model : `BertModel`
        预训练模型。

    # Returns

    adapters_groups : `nn.ModuleList`, required.
        所插入的所有 adapter, 用于绑定到模型中。
    """
    if not isinstance(transformer_model, (BertModel, ElectraModel, RobertaModel)):
        raise ConfigurationError("目前只支持 *BERT 结构")

    if isinstance(external_param, bool):
        param_place = [external_param for _ in range(adapter_layers)]
    elif isinstance(external_param, list):
        param_place = [False for _ in range(adapter_layers)]
        for i, e in enumerate(external_param, 1):
            param_place[-i] = e
    else:
        raise ConfigurationError("wrong type of external_param!")

    adapter_kwargs = adapter_kwargs or dict()
    adapter_kwargs.update(in_features=transformer_model.config.hidden_size)
    adapters_groups = nn.ModuleList([
        nn.ModuleList([
            Adapter(external_param=param_place[i], **adapter_kwargs),
            Adapter(external_param=param_place[i], **adapter_kwargs)
        ]) for i in range(adapter_layers)
    ])

    for i, adapters in enumerate(adapters_groups, 1):
        layer = transformer_model.encoder.layer[-i]
        layer.output = AdapterBertOutput(layer.output, adapters[0])
        layer.attention.output = AdapterBertOutput(layer.attention.output, adapters[1])

    return adapters_groups


"""
Adapter for transformers.
Parameter-Efficient Transfer Learning for NLP. https://arxiv.org/abs/1902.00751
"""

class Adapter(nn.Module):
    """
    Adapter module.
    """
    def __init__(self, in_features: int, adapter_size: int = 32, bias: bool = False,
                 activation: str = 'gelu', external_param: bool = False,
                 train_layer_norm: bool = True):
        super().__init__()
        self.in_features = in_features
        self.adapter_size = adapter_size
        self.bias = bias
        self.train_layer_norm = train_layer_norm
        self.act_fn = Activation.by_name(activation)()  # GELU is the best one.

        if external_param:
            self.weight_down, self.weight_up = None, None
        else:
            self.weight_down = nn.Parameter(torch.Tensor(adapter_size, in_features))
            self.weight_up = nn.Parameter(torch.Tensor(in_features, adapter_size))
            self.reset_parameters()

        if external_param or not bias:
            self.bias_down, self.bias_up = None, None
        else:
            self.bias_down = nn.Parameter(torch.zeros(adapter_size))
            self.bias_up = nn.Parameter(torch.zeros(in_features))

    def reset_parameters(self):
        nn.init.normal_(self.weight_down, std=1e-3)
        nn.init.normal_(self.weight_up, std=1e-3)

    def forward(self, hidden_states: torch.Tensor):
        linear_func = batched_linear if self.weight_down.dim() == 3 else linear
        x = linear_func(hidden_states, self.weight_down, self.bias_down)
        x = self.act_fn(x)
        x = linear_func(x, self.weight_up, self.bias_up)
        x = x + hidden_states
        return x


class AdapterBertOutput(nn.Module):
    """
    替代 BertOutput 和 BertSelfOutput
    """
    def __init__(self, base: Union[BertOutput, BertSelfOutput], adapter: Adapter):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter.forward
        for param in base.LayerNorm.parameters():
            param.requires_grad = adapter.train_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def batched_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    if b is not None:
        y = y + b.unsqueeze(1)
    return y

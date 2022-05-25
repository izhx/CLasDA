"""
Adapter transformers for AllenNLP.
Parameter-Efficient Transfer Learning for NLP. https://arxiv.org/abs/1902.00751
"""

from typing import Optional, Dict, Any, Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import linear

from transformers.models.bert.modeling_bert import BertModel, PreTrainedModel
from transformers.models.electra.modeling_electra import ElectraModel
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.funnel.modeling_funnel import FunnelBaseModel, INF
from transformers.models.vit.modeling_vit import ViTModel

from allennlp.common.checks import ConfigurationError
from allennlp.modules import Backbone, TokenEmbedder
from allennlp.nn import Activation

from .transformer_backbone import TransformerBackbone
from .transformer_embedder import TransformerEmbedder


class AdapterMixin:
    """
    目前只针对 *BERT 结构, 插入adapter.

    # Parameters

    layers : `int`, required.
        从 BERT 最后一层开始, 多少层插入adapter。
    kwargs : `Dict`, required.
        初始化 `Adapter` 的参数。
    external_param : `Union[bool, List[bool]]`
        adapter 的参数是否留空以便外部注入。

    """
    def __init__(
        self,
        layers: int = 6,
        kwargs: Optional[Dict[str, Any]] = None,
        external_param: Union[bool, List[bool]] = False
    ) -> None:
        if not isinstance(self._transformer_model, (
            BertModel, ElectraModel, RobertaModel, FunnelBaseModel, ViTModel
        )):
            raise ConfigurationError("Unsupported Model: %s", type(self._transformer_model))

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(layers)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(layers)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e
        else:
            raise ConfigurationError("wrong type of external_param!")

        kwargs = kwargs or dict()
        kwargs.update(in_features=self._transformer_model.config.hidden_size)
        self.adapters = nn.ModuleList([
            nn.ModuleList([
                Adapter(external_param=param_place[i], **kwargs),
                Adapter(external_param=param_place[i], **kwargs)
            ]) for i in range(layers)
        ])
        insert_adapters(self.adapters, self._transformer_model)

    def adapter_layers(self) -> int:
        return len(self.adapters)

    def adapter_size(self) -> int:
        return self.adapters[0][0].adapter_size

    def get_output_dim(self) -> int:
        raise NotImplementedError


@Backbone.register("adapter_transformer")
class AdapterTransformerBackbone(TransformerBackbone, AdapterMixin):
    def __init__(
        self,
        model_name: str,
        adapter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        super().__init__(model_name, train_parameters=False, **kwargs)
        AdapterMixin.__init__(self, **(adapter or {}))


@TokenEmbedder.register("adapter_transformer")
class AdapterTransformerEmbedder(TransformerEmbedder, AdapterMixin):
    def __init__(
        self,
        model_name: str,
        adapter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        super().__init__(model_name, train_parameters=False, **kwargs)
        AdapterMixin.__init__(self, **(adapter or {}))


def insert_adapters(adapters_groups: List[List['Adapter']], model: PreTrainedModel):
    """
    插入 adapters 到 BERT. 目前只支持 *BERT 结构!

    # Parameters
    adapters_groups: `List[List[Adapter]]`
        所有生成的 adapter
    model : `PreTrainedModel`
        预训练模型。
    """
    if isinstance(model, FunnelBaseModel):
        outer_index, inner_index = 1, 1
        for a, b in adapters_groups:
            layer = model.encoder.blocks[-outer_index][-inner_index]
            layer.ffn = AdapterFunnelPositionwiseFFN(layer.ffn, a)
            layer.attention = AdapterFunnelRelMultiheadAttention(layer.attention, b)
            if inner_index >= model.config.block_sizes[-outer_index]:
                outer_index, inner_index = outer_index + 1, 1
            else:
                inner_index += 1
    else:
        for i, adapters in enumerate(adapters_groups, 1):
            layer = model.encoder.layer[-i]
            if isinstance(model, ViTModel):
                layer.output = AdapterViTOutput(layer.output, adapters[0])
                model.encoder.layer[-i] = AdapterViTLayer(layer, adapters[1])
            else:
                layer.output = AdapterBertOutput(layer.output, adapters[0])
                layer.attention.output = AdapterBertOutput(
                    layer.attention.output, adapters[1]
                )
    return


class Adapter(nn.Module):
    """
    Adapter module.
    Parameter-Efficient Transfer Learning for NLP. https://arxiv.org/abs/1902.00751
    """
    def __init__(
        self,
        in_features: int,
        adapter_size: int = 32,
        bias: bool = False,
        activation: str = 'gelu',
        external_param: bool = False,
        train_layer_norm: bool = True
    ):
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


class AdapterProxy(nn.Module):
    """
    替代要插入 `Adapter` 的 modules, such as `BertOutput` and `BertSelfOutput`.
    """
    def __init__(self, base: nn.Module, adapter: Adapter):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter.forward
        self.set_layer_norm(adapter.train_layer_norm)

    def set_layer_norm(self, requires_grad: bool):
        for name, module in self.base.named_modules():
            if name.lower().startswith(("layernorm", "layer_norm")):
                for param in module.parameters():
                    param.requires_grad = requires_grad


class AdapterBertOutput(AdapterProxy):
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)  # Adapter
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AdapterFunnelRelMultiheadAttention(AdapterProxy):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_inputs: Tuple[torch.Tensor],
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = self.base.config.n_head, self.base.config.d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = self.base.q_head(query).view(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = self.base.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = self.base.v_head(value).view(batch_size, context_len, n_head, d_head)

        q_head = q_head * self.base.scale
        # Shape n_head x d_head
        r_w_bias = self.base.r_w_bias * self.base.scale
        # Shapes batch_size x n_head x seq_len x context_len
        content_score = torch.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        positional_attn = self.base.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.base.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn

        # precision safe in case of mixed precision training
        dtype = attn_score.dtype
        attn_score = attn_score.float()
        # perform masking
        if attention_mask is not None:
            attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
        # attention probability
        attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
        attn_prob = self.base.attention_dropout(attn_prob)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = torch.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.base.post_proj(attn_vec.reshape(batch_size, seq_len, n_head * d_head))
        attn_out = self.base.hidden_dropout(attn_out)

        attn_out = self.adapter_forward(attn_out)  # only insert this line!

        output = self.base.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class AdapterFunnelPositionwiseFFN(AdapterProxy):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.base.linear_1(hidden)
        h = self.base.activation_function(h)
        h = self.base.activation_dropout(h)
        h = self.base.linear_2(h)
        h = self.base.dropout(h)
        h = self.adapter_forward(h)  # Adapter
        return self.base.layer_norm(hidden + h)


class AdapterViTOutput(AdapterProxy):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)  # Adapter
        hidden_states = hidden_states + input_tensor
        return hidden_states


class AdapterViTLayer(AdapterProxy):
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.base.attention(
            self.base.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        attention_output = self.adapter_forward(attention_output)  # Adapter

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.base.layernorm_after(hidden_states)
        layer_output = self.base.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.base.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


def batched_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    if b is not None:
        y = y + b.unsqueeze(1)
    return y

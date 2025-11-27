# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import torch

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..flex_olmo.configuration_flex_olmo import FlexOlmoConfig
from ..flex_olmo.modeling_flex_olmo import (
    FlexOlmoAttention,
    FlexOlmoDecoderLayer,
    FlexOlmoForCausalLM,
    FlexOlmoModel,
    FlexOlmoPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class FlexOlmoNoQKNormPrenormConfig(FlexOlmoConfig):
    model_type = "flex_olmo_noqknorm_prenorm"
    # Update base_model_tp_plan to remove the "rep" suffixes since no qk-norms
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",  # No longer need rep
        "layers.*.self_attn.k_proj": "colwise",  # No longer need rep
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",  # No longer need rep
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }


class FlexOlmoNoQKNormPrenormAttention(FlexOlmoAttention):
    def __init__(self, config: FlexOlmoNoQKNormPrenormConfig, layer_idx: Optional[int] = None):
        super().__init__(
            config=config,
            layer_idx=layer_idx,
        )
        # we remove qk norm
        del self.q_norm
        del self.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def FlexOlmoNoQKNormPrenormRMSNorm(FlexOlmoRMSNorm):
    pass


class FlexOlmoNoQKNormPrenormDecoderLayer(FlexOlmoDecoderLayer):
    def __init__(self, config: FlexOlmoNoQKNormPrenormConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.post_attention_layernorm
        del self.post_feedforward_layernorm

        self.pre_attention_layernorm = FlexOlmoNoQKNormPrenormRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = FlexOlmoNoQKNormPrenormRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        # apply norm before attention
        hidden_states = self.pre_attention_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # apply norm before feedforward
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class FlexOlmoNoQKNormPrenormPreTrainedModel(FlexOlmoPreTrainedModel):
    config_class = FlexOlmoNoQKNormPrenormConfig

class FlexOlmoNoQKNormPrenormModel(FlexOlmoModel):
    pass

class FlexOlmoNoQKNormPrenormForCausalLM(FlexOlmoForCausalLM):
    pass


__all__ = [
    "FlexOlmoNoQKNormPrenormConfig",
    "FlexOlmoNoQKNormPrenormForCausalLM",
    "FlexOlmoNoQKNormPrenormModel",
    "FlexOlmoNoQKNormPrenormPreTrainedModel",
]

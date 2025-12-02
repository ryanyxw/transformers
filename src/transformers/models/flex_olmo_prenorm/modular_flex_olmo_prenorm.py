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

from typing import Optional

import torch

from ...cache_utils import Cache
from ..flex_olmo.configuration_flex_olmo import FlexOlmoConfig
from ..flex_olmo.modeling_flex_olmo import (
    FlexOlmoAttention,
    FlexOlmoDecoderLayer,
    FlexOlmoForCausalLM,
    FlexOlmoModel,
    FlexOlmoPreTrainedModel,
    FlexOlmoRMSNorm,
)


class FlexOlmoPrenormConfig(FlexOlmoConfig):
    model_type = "flex_olmo_prenorm"


class FlexOlmoPrenormRMSNorm(FlexOlmoRMSNorm):
    pass


class FlexOlmoPrenormAttention(FlexOlmoAttention):
    pass


class FlexOlmoPrenormDecoderLayer(FlexOlmoDecoderLayer):
    def __init__(self, config: FlexOlmoPrenormConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.post_attention_layernorm
        del self.post_feedforward_layernorm

        self.pre_attention_layernorm = FlexOlmoPrenormRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = FlexOlmoPrenormRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class FlexOlmoPrenormPreTrainedModel(FlexOlmoPreTrainedModel):
    config_class = FlexOlmoPrenormConfig


class FlexOlmoPrenormModel(FlexOlmoModel):
    pass


class FlexOlmoPrenormForCausalLM(FlexOlmoForCausalLM):
    pass


__all__ = [
    "FlexOlmoPrenormConfig",
    "FlexOlmoPrenormForCausalLM",
    "FlexOlmoPrenormModel",
    "FlexOlmoPrenormPreTrainedModel",
]

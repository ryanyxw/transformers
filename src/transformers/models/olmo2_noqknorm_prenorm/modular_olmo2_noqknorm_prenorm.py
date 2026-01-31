from typing import Callable, Optional

import torch

from transformers.utils.generic import TransformersKwargs

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import eager_attention_forward
from ..olmo2.configuration_olmo2 import Olmo2Config
from ..olmo2.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2Model,
    Olmo2PreTrainedModel,
    Olmo2RMSNorm,
    apply_rotary_pos_emb,
)


logger = logging.get_logger(__name__)


class Olmo2NoQKNormPrenormConfig(Olmo2Config):
    model_type = "olmo2_noqknorm_prenorm"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",  # No longer need rep
        "layers.*.self_attn.k_proj": "colwise",  # No longer need rep
        "layers.*.self_attn.v_proj": "colwise",  # No longer need rep
        "layers.*.self_attn.o_proj": "rowwise",  # No longer need rep
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }


class Olmo2NoQKNormPrenormRMSNorm(Olmo2RMSNorm):
    pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Olmo2NoQKNormPrenormAttention(Olmo2Attention):
    def __init__(self, config: Olmo2NoQKNormPrenormConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
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


class Olmo2NoQKNormPrenormDecoderLayer(Olmo2DecoderLayer):
    def __init__(self, config: Olmo2NoQKNormPrenormConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        del self.post_attention_layernorm
        del self.post_feedforward_layernorm

        self.pre_attention_layernorm = Olmo2NoQKNormPrenormRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Olmo2NoQKNormPrenormRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        # apply norm before attention
        hidden_states = self.pre_attention_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        # apply norm before feedforward
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class Olmo2NoQKNormPrenormPreTrainedModel(Olmo2PreTrainedModel):
    pass


# The OLMo2 model is identical to the OLMo model, except RMSNorm is used instead of
# standard layer norm for the output norm.
class Olmo2NoQKNormPrenormModel(Olmo2Model):
    pass


# The heads now only need to redefine the model inside to the correct `RobertaModel`
class Olmo2NoQKNormPrenormForCausalLM(Olmo2ForCausalLM):
    pass


__all__ = [
    "Olmo2NoQKNormPrenormConfig",
    "Olmo2NoQKNormPrenormForCausalLM",
    "Olmo2NoQKNormPrenormModel",
    "Olmo2NoQKNormPrenormPreTrainedModel",
]

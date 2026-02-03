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

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring
from ..flex_olmo.configuration_flex_olmo import FlexOlmoConfig
from ..flex_olmo.modeling_flex_olmo import (
    FlexOlmoAttention,
    FlexOlmoDecoderLayer,
    FlexOlmoForCausalLM,
    FlexOlmoModel,
    FlexOlmoPreTrainedModel,
    FlexOlmoRMSNorm,
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


class FlexOlmoNoQKNormPrenormRMSNorm(FlexOlmoRMSNorm):
    pass


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


@dataclass
class MoeCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).

        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.

        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.

        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    lb_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None


def load_balancing_loss_func_olmoe(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[
        torch.Tensor
    ] = None,  # the number of tokens within a global batch (including across dp ranks)
    ignore_index=-100,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts]. This has not been softmaxed yet
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.stack(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )  # shape: (num_hidden_layers, batch_size * sequence_length, num_experts)
        # concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0) # shape: (num_hidden_layers * batch_size * sequence_length, num_experts)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(
        routing_weights, top_k, dim=-1
    )  # shape: (num_hidden_layers, batch_size * sequence_length, top_k)

    expert_counts_onehot = torch.nn.functional.one_hot(
        selected_experts, num_experts
    )  # shape: (num_hidden_layers, batch_size * sequence_length, top_k, num_experts)

    if attention_mask is None and labels is None:
        # Compute the percentage of tokens routed to each experts
        counts_per_expert = torch.mean(
            expert_counts_onehot.float(), dim=(1, 2)
        )  # shape: (num_hidden_layers, num_experts)

        # Compute the average probability of routing to these experts
        prob_per_expert = torch.mean(routing_weights, dim=1)  # shape: (num_hidden_layers, num_experts)
    else:
        # if there are labels, then we want to ignore the indices that are in the prompt as well (if there is any)
        if labels is not None:
            attention_mask = labels != ignore_index
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0]

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(num_hidden_layers, -1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        # frequency_per_expert = torch.sum(expert_counts_onehot.float() * expert_attention_mask, dim=(1, 2)) / torch.sum(expert_attention_mask, dim=(1,2)) # shape: (num_hidden_layers, num_experts)
        counts_per_expert = torch.sum(expert_counts_onehot.float() * expert_attention_mask, dim=(1, 2))

        # Compute the mask that masks all padding tokens as 0 with the same shape of frequency_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(num_hidden_layers, -1, num_experts)
            .to(compute_device)
        )

        # average the probability across valid tokens
        prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=1) / torch.sum(
            attention_mask
        )  # shape: (num_hidden_layers, num_experts)

    overall_loss = torch.sum(counts_per_expert * prob_per_expert)

    # Fallback when num_items_in_batch isn't provided (e.g., manual forward calls)
    if num_items_in_batch is None:
        if labels is not None:
            num_items_in_batch = (labels != ignore_index).sum()
        elif attention_mask is not None:
            num_items_in_batch = attention_mask.sum()
        else:
            # fall back to total tokens in batch/seq from gate logits
            num_items_in_batch = gate_logits[0].shape[0]

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(compute_device)

    # we follow olmo-core and use counts for dot product instead of frequency, and divide by total number token across gradient accumulation steps
    overall_loss = overall_loss / (num_items_in_batch * top_k)

    overall_loss = (
        overall_loss * num_experts / concatenated_gate_logits.shape[0]
    )  # times num_experts according to lb equation, divide by num_hidden_layers to get average over layers (which is how olmo-core is implemented to make loss agnostic to number of layers)

    return overall_loss


class FlexOlmoNoQKNormPrenormForCausalLM(FlexOlmoForCausalLM):
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlexOlmoNoQKNormPrenormForCausalLM

        >>> model = FlexOlmoNoQKNormPrenormForCausalLM.from_pretrained("allenai/FlexOlmoNoQKNormPrenorm-1B-7B-0924")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/FlexOlmoNoQKNormPrenorm-1B-7B-0924")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        ce_loss = None
        if labels is not None:
            ce_loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
            loss = ce_loss

        lb_loss = None

        if output_router_logits:
            lb_loss = load_balancing_loss_func_olmoe(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
                labels,
                **kwargs,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * lb_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (lb_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=lb_loss,
            lb_loss=lb_loss.detach().clone() if lb_loss is not None else None,  # for logging callback
            ce_loss=ce_loss.detach().clone() if ce_loss is not None else None,  # for logging callback
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "FlexOlmoNoQKNormPrenormConfig",
    "FlexOlmoNoQKNormPrenormForCausalLM",
    "FlexOlmoNoQKNormPrenormModel",
    "FlexOlmoNoQKNormPrenormPreTrainedModel",
]

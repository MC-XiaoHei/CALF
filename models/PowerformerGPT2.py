"""
Custom GPT2 architecture with Powerformer's Weighted Causal Multihead Attention.

This module provides GPT2 blocks that use WCMHA instead of standard self-attention
for improved time series modeling with power-law decay temporal dependencies.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from models.PowerformerAttention import WeightedCausalMultiheadAttention


class PowerformerGPT2Attention(nn.Module):
    """
    GPT2 Attention module using WCMHA instead of standard attention.
    This is a drop-in replacement for GPT2Attention that uses power-law decay.
    """
    
    def __init__(self, config, alpha: float = 1.0, is_cross_attention: bool = False, layer_idx: Optional[int] = None):
        super().__init__()
        
        self.config = config
        self.is_cross_attention = is_cross_attention
        self.layer_idx = layer_idx
        
        max_positions = config.max_position_embeddings
        self.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
            1, 1, max_positions, max_positions
        )
        self.masked_bias = torch.tensor(-1e4)
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        # Use WCMHA for self-attention
        if not is_cross_attention:
            self.wcmha = WeightedCausalMultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                alpha=alpha,
                dropout=config.attn_pdrop,
                bias=True,
            )
            # For compatibility, we still need these attributes even though WCMHA handles them
            self.c_attn = self.wcmha.c_attn
            self.c_proj = self.wcmha.c_proj
        else:
            # For cross-attention, use standard projections
            self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim)
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
            self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass using WCMHA for self-attention.
        """
        if encoder_hidden_states is not None:
            # This is cross-attention, use standard attention mechanism
            if not hasattr(self, 'q_attn'):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `PowerformerGPT2Attention(..., is_cross_attention=True)`."
                )
            
            # Standard cross-attention implementation
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
            
            # Implement standard attention here
            # This is a simplified version - full implementation would match GPT2Attention
            raise NotImplementedError("Cross-attention with WCMHA is not yet implemented")
        else:
            # Self-attention using WCMHA
            outputs = self.wcmha(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            return outputs


class PowerformerGPT2Block(nn.Module):
    """
    GPT2 Block using Powerformer's WCMHA attention.
    This is a modified version of GPT2Block that uses PowerformerGPT2Attention.
    """
    
    def __init__(self, config, alpha: float = 1.0, layer_idx: Optional[int] = None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = PowerformerGPT2Attention(config, alpha=alpha, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        
        if config.add_cross_attention:
            self.crossattention = PowerformerGPT2Attention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        
        # MLP
        self.mlp = nn.ModuleDict({
            'c_fc': nn.Linear(hidden_size, inner_dim),
            'c_proj': nn.Linear(inner_dim, hidden_size),
            'act': nn.GELU(),
            'dropout': nn.Dropout(config.resid_pdrop),
        })
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for the Powerformer GPT2 block.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        # Residual connection
        hidden_states = attn_output + residual
        
        # Cross-attention (if applicable)
        if encoder_hidden_states is not None:
            # Add cross-attention if model has it
            if not hasattr(self, 'crossattention'):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp['c_fc'](hidden_states)
        feed_forward_hidden_states = self.mlp['act'](feed_forward_hidden_states)
        feed_forward_hidden_states = self.mlp['c_proj'](feed_forward_hidden_states)
        feed_forward_hidden_states = self.mlp['dropout'](feed_forward_hidden_states)
        
        # Residual connection
        hidden_states = residual + feed_forward_hidden_states
        
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        
        return outputs  # hidden_states, present, (attentions, cross_attentions)


class PowerformerGPT2Model(GPT2Model):
    """
    GPT2 Model that uses Powerformer's WCMHA for time series modeling.
    This model replaces standard GPT2 blocks with PowerformerGPT2Block.
    """
    
    def __init__(self, config, alpha: float = 1.0):
        super().__init__(config)
        self.alpha = alpha
        
        # Replace the standard GPT2 blocks with Powerformer blocks
        self.h = nn.ModuleList([
            PowerformerGPT2Block(config, alpha=alpha, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Re-initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Forward pass - delegates to parent GPT2Model which uses our custom blocks.
        """
        # The parent class forward method will use our custom self.h blocks
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

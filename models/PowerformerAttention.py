import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class WeightedCausalMultiheadAttention(nn.Module):
    """
    Weighted Causal Multihead Attention (WCMHA) from Powerformer.
    
    This module implements power-law decay attention for time series:
    - Causal masking: future cannot attend to past (mask = -inf)
    - Power-law decay: attention decays as -alpha * log(delta_t) for causal connections
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        alpha: Power-law decay parameter (controls decay speed)
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V (combined for efficiency like GPT2)
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def _create_power_law_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Create power-law decay mask for causal attention.
        
        Args:
            seq_len: Sequence length
            device: Device for the mask tensor
            dtype: Data type for the mask tensor
            
        Returns:
            Mask tensor of shape (seq_len, seq_len)
        """
        # Create time difference matrix: delta_t[i,j] = i - j
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        delta_t = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
        
        # Initialize mask
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        
        # For non-causal connections (delta_t < 0, i.e., future attending to current/past)
        # Set to -inf so attention weight becomes 0
        mask = mask.masked_fill(delta_t < 0, float('-inf'))
        
        # For causal connections (delta_t >= 0), apply power-law decay: -alpha * log(delta_t)
        # Note: delta_t = 0 (self-attention) should have no penalty
        # For delta_t > 0, apply the power-law decay
        causal_mask = delta_t > 0
        if causal_mask.any():
            # Add small epsilon to avoid log(0), then apply power-law decay
            log_delta_t = torch.log(delta_t.float() + 1.0)  # log(delta_t + 1) to handle delta_t=0
            power_law_penalty = -self.alpha * log_delta_t
            mask = torch.where(causal_mask, power_law_penalty.to(dtype), mask)
        
        return mask
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dimension into num_heads and head_dim.
        
        Args:
            tensor: Input tensor of shape (batch, seq_len, embed_dim)
            
        Returns:
            Tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
    
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge num_heads and head_dim back into embed_dim.
        
        Args:
            tensor: Input tensor of shape (batch, num_heads, seq_len, head_dim)
            
        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_heads, head_dim)
        return tensor.view(batch_size, seq_len, self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for WCMHA.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask
            layer_past: Optional cached key/value from previous forward pass
            head_mask: Optional mask for attention heads
            use_cache: Whether to return key/value for caching
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, present, attentions) where:
                - output: Attention output of shape (batch, seq_len, embed_dim)
                - present: Optional cached (key, value) if use_cache=True
                - attentions: Optional attention weights if output_attentions=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=2)
        
        # Split heads
        query = self._split_heads(query)  # (batch, num_heads, seq_len, head_dim)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # Handle cached key/value for generation
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        if use_cache:
            present = (key, value)
        else:
            present = None
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))  # (batch, num_heads, seq_len, key_seq_len)
        attn_weights = attn_weights * self.scaling
        
        # Get the sequence length for the keys (might be longer than seq_len if using cache)
        key_seq_len = key.shape[-2]
        
        # Apply power-law decay mask
        power_law_mask = self._create_power_law_mask(
            key_seq_len, 
            device=hidden_states.device, 
            dtype=attn_weights.dtype
        )
        
        # If we have past keys, we only apply mask to the relevant portion
        if layer_past is not None:
            # For generation, only apply mask to new positions
            power_law_mask = power_law_mask[-seq_len:, :]
        
        # Add power-law mask (broadcasting across batch and heads)
        attn_weights = attn_weights + power_law_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply additional attention mask if provided (e.g., padding mask)
        if attention_mask is not None:
            # attention_mask shape: (batch, 1, 1, key_seq_len) or similar
            attn_weights = attn_weights + attention_mask
        
        # Softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        # Compute attention output: weighted sum of values
        attn_output = torch.matmul(attn_weights, value)  # (batch, num_heads, seq_len, head_dim)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output)  # (batch, seq_len, embed_dim)
        
        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        
        return outputs  # (attn_output, present, (attentions))

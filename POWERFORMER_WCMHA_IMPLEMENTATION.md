# Powerformer WCMHA Implementation for CALF

## Overview

This implementation adds Powerformer's Weighted Causal Multihead Attention (WCMHA) to the CALF model's time series branch to address the "global reverberation" issue in standard self-attention mechanisms.

## Problem Statement

Time series data follows different generation patterns than natural language:

1. **Causality**: Past influences future, but future cannot influence past (unidirectional time flow)
2. **Temporal Locality/Decaying Influence**: Recent events have stronger impact; distant past events have decaying influence

Standard self-attention violates these principles by:
- Allowing bidirectional information flow (violates causality)
- Giving equal initial attention to all time distances (violates locality)

## Solution: Powerformer's WCMHA

WCMHA modifies attention scores with a carefully designed mask:

- **Non-causal connections** (future → past): mask = -∞ → attention weight = 0
- **Causal connections** (past → future): mask = -α·log(Δt) where:
  - Δt = time difference between query and key
  - α = hyperparameter controlling decay speed

The power-law decay provides:
- Strong "focus on recent" inductive bias
- Preservation of ability to capture important long-term dependencies
- Better match to real-world time series characteristics

## Files Modified/Created

### 1. `/models/PowerformerAttention.py` (NEW)
Implements the core WCMHA mechanism:

```python
class WeightedCausalMultiheadAttention(nn.Module):
    """
    Implements power-law decay attention for time series.
    
    Key features:
    - Causal masking: future cannot attend to past
    - Power-law decay: -α·log(Δt + 1) for causal connections
    - Compatible with GPT2 attention interface
    """
```

**Key Methods:**
- `_create_power_law_mask()`: Creates the power-law decay mask
- `forward()`: Applies WCMHA with proper caching support

**Dimension Handling:**
- Input: (batch, seq_len, embed_dim)
- Output: (batch, seq_len, embed_dim)
- Attention weights: (batch, num_heads, seq_len, seq_len)
- Proper handling of cached keys/values for generation

### 2. `/models/PowerformerGPT2.py` (NEW)
Provides GPT2 blocks that use WCMHA:

```python
class PowerformerGPT2Attention(nn.Module):
    """Drop-in replacement for GPT2Attention using WCMHA"""

class PowerformerGPT2Block(nn.Module):
    """GPT2 block with WCMHA attention"""

class PowerformerGPT2Model(GPT2Model):
    """Full GPT2 model using Powerformer blocks"""
```

**Architecture:**
- Replaces standard self-attention with WCMHA
- Preserves layer norms and MLP structures
- Compatible with PEFT/LoRA fine-tuning

### 3. `/models/GPT2_arch.py` (MODIFIED)
Added `PowerformerAccustumGPT2Model` class:

```python
class PowerformerAccustumGPT2Model(GPT2Model):
    """
    Custom GPT2 model with Powerformer's WCMHA for time series.
    Includes custom forward pass compatible with CALF architecture.
    """
```

**Changes:**
- Imports `PowerformerGPT2Block`
- Implements full forward pass with WCMHA blocks
- Maintains compatibility with original `AccustumGPT2Model` interface

### 4. `/models/CALF.py` (MODIFIED)
Updated to use WCMHA for time branch:

```python
class Model(nn.Module):
    def __init__(self, configs, device):
        # ...
        
        # NEW: Get alpha parameter from configs
        alpha = getattr(configs, 'powerformer_alpha', 1.0)
        
        # NEW: Use PowerformerAccustumGPT2Model for time branch
        base_gpt2_time = AccustumGPT2Model.from_pretrained('gpt2', ...)
        self.gpt2 = PowerformerAccustumGPT2Model(base_gpt2_time.config, alpha=alpha)
        
        # Copy pretrained weights (except attention)
        # ... weight copying logic ...
        
        # Text branch uses standard GPT2 (unchanged)
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', ...)
```

**Key Changes:**
1. Added import: `PowerformerAccustumGPT2Model`
2. Added alpha parameter support via `configs.powerformer_alpha`
3. Time branch (`self.gpt2`) now uses WCMHA
4. Text branch (`self.gpt2_text`) remains unchanged
5. Proper weight initialization from pretrained GPT2:
   - Token/position embeddings copied
   - Layer norms copied
   - MLP weights copied
   - WCMHA attention randomly initialized (will be fine-tuned)

## Configuration Parameter

Add to your config file:

```python
powerformer_alpha = 1.0  # Power-law decay parameter
```

**Typical values:**
- `alpha = 0.5`: Weak decay (more long-term dependencies)
- `alpha = 1.0`: Moderate decay (default, balanced)
- `alpha = 2.0`: Strong decay (focus on recent)

## Dimension Safety

All implementations include careful dimension handling to avoid errors:

1. **Mask Creation:**
   - Uses `torch.float32` for intermediate calculations
   - Converts to target dtype only at the end
   - Properly handles log(0) with `log(Δt + 1)`

2. **Attention Computation:**
   - Proper splitting/merging of attention heads
   - Correct broadcasting of masks across batches and heads
   - Cache-aware mask slicing for generation

3. **NoneType Safety:**
   - All optional parameters checked before use
   - Proper default values for `layer_past`, `attention_mask`, `head_mask`

## Testing Recommendations

1. **Basic Forward Pass:**
   ```python
   model = Model(configs, device)
   x = torch.randn(batch_size, seq_len, enc_in)
   output = model(x)
   ```

2. **Check Dimensions:**
   - Verify output shapes match expected dimensions
   - Test with various batch sizes and sequence lengths

3. **Verify Causality:**
   - Extract attention weights with `output_attentions=True`
   - Verify upper triangle is near-zero (causal masking)

4. **Power-law Decay:**
   - Extract attention weights
   - Verify decay pattern: stronger attention to recent tokens

## Expected Benefits

1. **Better Generalization:** Model learns appropriate temporal inductive bias
2. **Data Efficiency:** Requires less data to learn time series patterns
3. **Robustness:** Less prone to spurious long-distance correlations
4. **Physical Intuition:** Matches real-world time series characteristics

## Backward Compatibility

- Text branch remains unchanged (standard GPT2)
- All existing functionality preserved
- Only time branch uses WCMHA
- Compatible with existing training scripts
- Works with PEFT/LoRA fine-tuning

## Implementation Notes

1. **Weight Initialization:**
   - WCMHA attention weights are randomly initialized
   - Other components (embeddings, norms, MLPs) copied from pretrained GPT2
   - This is intentional: WCMHA needs to learn power-law patterns

2. **PEFT/LoRA Compatibility:**
   - LoRA applied AFTER creating Powerformer model
   - Target modules include WCMHA's `c_attn` projection
   - Fine-tuning will adapt WCMHA to specific time series patterns

3. **Gradient Flow:**
   - Only specific parameters trainable (controlled by existing logic)
   - WCMHA parameters included in trainable set via LoRA

## Future Improvements

Potential enhancements:
1. Learnable alpha parameter (per-layer or per-head)
2. Different decay functions (exponential, polynomial)
3. Adaptive decay based on data characteristics
4. Cross-attention with power-law decay

## References

- Powerformer paper: [Add reference if available]
- Original CALF implementation
- GPT2 architecture (Hugging Face transformers)

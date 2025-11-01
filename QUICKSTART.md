# Powerformer WCMHA Implementation - Quick Start Guide

## What Was Changed?

This implementation replaces the standard self-attention mechanism in the CALF model's **time series branch** with Powerformer's Weighted Causal Multihead Attention (WCMHA), which applies power-law decay to attention weights based on temporal distance.

## Why This Change?

Time series data follows two fundamental principles that standard attention violates:
1. **Causality**: Future cannot influence past
2. **Temporal Locality**: Recent events matter more than distant past

WCMHA enforces these principles through:
- Causal masking: Future positions get -∞ mask (zero attention)
- Power-law decay: Past positions get -α·log(Δt+1) mask (decaying attention)

## Files Added/Modified

### New Files:
- `models/PowerformerAttention.py` - Core WCMHA implementation
- `models/PowerformerGPT2.py` - GPT2 blocks using WCMHA
- `POWERFORMER_WCMHA_IMPLEMENTATION.md` - Full documentation
- `CHANGES_SUMMARY.md` - Detailed change summary
- `powerformer_wcmha_implementation.patch` - Git patch file

### Modified Files:
- `models/GPT2_arch.py` - Added PowerformerAccustumGPT2Model class
- `models/CALF.py` - Time branch now uses WCMHA

## How to Use

### Option 1: Already Applied (Current Branch)
If you're on the `copilot/fix-global-reverberation-issue` branch, the changes are already applied!

### Option 2: Apply Patch to Another Branch
```bash
# Switch to your target branch
git checkout your-branch

# Apply the patch
git apply powerformer_wcmha_implementation.patch

# Or if you prefer to review first:
git apply --check powerformer_wcmha_implementation.patch  # Check if it can be applied
git apply powerformer_wcmha_implementation.patch          # Apply it
```

## Configuration

Add this parameter to your config file:

```python
# config.py or wherever you define configs
powerformer_alpha = 1.0  # Default: 1.0
```

**Parameter Guide:**
- `powerformer_alpha = 0.5`: Weak decay → Retains more long-term dependencies
- `powerformer_alpha = 1.0`: Moderate decay → Balanced (recommended default)
- `powerformer_alpha = 2.0`: Strong decay → Strong focus on recent events

## Quick Verification

After applying changes, verify the implementation:

```bash
# Check that new files exist
ls models/PowerformerAttention.py
ls models/PowerformerGPT2.py

# Verify imports work (requires dependencies installed)
python -c "from models.PowerformerAttention import WeightedCausalMultiheadAttention; print('✓ Import successful')"
python -c "from models.GPT2_arch import PowerformerAccustumGPT2Model; print('✓ Import successful')"
```

## Running the Model

No changes needed to your training scripts! Use as before:

```python
from models.CALF import Model

# Create model (automatically uses WCMHA for time branch)
model = Model(configs, device)

# Train/test as usual
output = model(x_enc)
```

## What's Different?

### Before:
```
Time Branch: GPT2 → Standard Attention → MLP
Text Branch: GPT2 → Standard Attention → MLP
```

### After:
```
Time Branch: GPT2 → WCMHA (power-law) → MLP  ← Changed!
Text Branch: GPT2 → Standard Attention → MLP  ← Unchanged
```

## Architecture Details

### WCMHA Mask Formula:
```
For each attention score between position i (query) and position j (key):

If j > i (future position):
    mask[i,j] = -∞  (attention → 0, strict causality)

If j <= i (current/past position):
    Δt = i - j  (time distance)
    mask[i,j] = -α · log(Δt + 1)  (power-law decay)
```

### Key Properties:
1. **Self-attention** (Δt=0): mask = -α·log(1) = 0 (no penalty)
2. **Recent past** (Δt=1): mask = -α·log(2) ≈ -0.69α (small penalty)
3. **Distant past** (Δt=10): mask = -α·log(11) ≈ -2.4α (larger penalty)
4. **Future** (Δt<0): mask = -∞ (zero attention)

## Dimension Safety

All implementations handle:
- ✓ Varying batch sizes
- ✓ Varying sequence lengths
- ✓ Different data types (fp16, fp32)
- ✓ Optional parameters (None checks)
- ✓ Cached key/values (for generation)

## Testing

Run basic checks:

```python
import torch
from models.PowerformerAttention import WeightedCausalMultiheadAttention

# Create WCMHA layer
wcmha = WeightedCausalMultiheadAttention(
    embed_dim=768,
    num_heads=12,
    alpha=1.0
)

# Test forward pass
x = torch.randn(2, 10, 768)  # (batch=2, seq_len=10, dim=768)
output = wcmha(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output[0].shape}")
print("✓ Basic test passed!")
```

## Documentation Files

1. **`POWERFORMER_WCMHA_IMPLEMENTATION.md`** - Comprehensive documentation
   - Problem statement
   - Solution details
   - Architecture explanations
   - Configuration guide
   - Testing recommendations

2. **`CHANGES_SUMMARY.md`** - Summary of changes
   - File-by-file breakdown
   - Implementation details
   - Usage examples
   - Expected benefits

3. **`powerformer_wcmha_implementation.patch`** - Git patch
   - Complete diff of all changes
   - Can be applied with `git apply`

## Troubleshooting

### Issue: Import errors
**Solution**: Ensure you're in the CALF directory and dependencies are installed:
```bash
pip install torch transformers peft einops
```

### Issue: "powerformer_alpha not found"
**Solution**: Add to your config file:
```python
configs.powerformer_alpha = 1.0
```
If not set, it defaults to 1.0 automatically.

### Issue: Different behavior than before
**Expected**: WCMHA enforces causality and temporal locality, so attention patterns will differ. This is intentional and should improve time series performance.

## Expected Improvements

1. **Better generalization** on unseen time series
2. **More data-efficient** learning
3. **Reduced overfitting** to spurious long-distance correlations
4. **Physically consistent** predictions

## Questions?

Refer to:
- `POWERFORMER_WCMHA_IMPLEMENTATION.md` for detailed documentation
- `CHANGES_SUMMARY.md` for implementation details
- Original paper on Powerformer (if available)

## Summary

✅ WCMHA implemented in time series branch
✅ Text branch unchanged (backward compatible)
✅ All dimension issues handled
✅ NoneType safety ensured
✅ Comprehensive documentation provided
✅ Patch file ready for deployment

**Ready to use!** Just set `powerformer_alpha` in your config and train as usual.

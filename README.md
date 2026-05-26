[zamba2-README.md](https://github.com/user-attachments/files/28271516/zamba2-README.md)
# Zamba2-7B SLERP Merge: Weight-Sharing Breaks Standard Merge Tooling

**Second experiment in a series testing SLERP merging across non-transformer architectures.**

This repo documents an attempt to SLERP merge [Zyphra/Zamba2-7B](https://huggingface.co/Zyphra/Zamba2-7B) (base) and [Zyphra/Zamba2-7B-Instruct](https://huggingface.co/Zyphra/Zamba2-7B-Instruct) using the same methodology as our [Falcon-H1 SLERP merge](https://github.com/iamboosted/falcon-h1-slerp-merge).

**Result: The merge itself succeeds, but the merged model cannot be loaded for evaluation.** Zamba2's shared-weight architecture is fundamentally incompatible with standard SLERP merge + eval tooling. This is a more interesting finding than a benchmark number.

## Context: The Broader Study

This is part of a series testing whether SLERP works on non-transformer architectures:

| Architecture | Model | Hybrid Type | SLERP Merge | Eval |
|---|---|---|---|---|
| Falcon-H1 | 7B-Instruct × H1R-7B | Parallel (attn + Mamba in same block) | ✅ Functional | ✅ 65% (degraded from 80%/70% parents) |
| **Zamba2** | **7B × 7B-Instruct** | **Sequential (Mamba backbone + shared attn)** | **✅ Completes** | **❌ Cannot load** |

The question was: does the hybrid design affect whether SLERP works? The answer is yes, but not in the way we expected. Zamba2 doesn't degrade gracefully like Falcon-H1 — it can't even be evaluated because the architecture's weight-sharing mechanism is incompatible with standard tooling.

## Why Zamba2 Breaks

Zamba2's architecture is fundamentally different from Falcon-H1 (and from standard transformers):

**Shared transformer blocks with position-specific LoRA adapters.** Zamba2 has only 2 attention blocks for the entire 81-layer model. These blocks are *shared* — the same weights are reused at every position they appear (layers 6, 17, 28, 39, 50, 61, 72 for block A, etc.). To give each position slight specialization, Zyphra bakes LoRA adapters directly into the architecture: even-indexed adapters (0, 2, 4, 6, 8, 10, 12) belong to one shared block, odd-indexed (1, 3, 5, 7, 9, 11) to the other.

In the safetensors files, the shared weights are stored **once** and referenced via a weight-tying mechanism at load time. SLERP merging creates independent copies at each layer position, which breaks this tying in two ways:

1. **Validation failure:** HuggingFace transformers' `get_expanded_tied_weights_keys()` expects the tied weight sets to have matching sub-parameter names. Because the adapter indices differ (even vs odd), validation fails with a `ValueError`.

2. **Weight loading failure:** Patching around the validation causes the shared weights at layers 17-77 to load as MISSING (randomly initialized), because the tying mechanism that's supposed to copy them from layer 6 never executes.

3. **Quantization failure:** Even if weights load, bitsandbytes 4-bit quantization chokes on the improperly initialized shared layers with an `AssertionError` on weight shape validation.

The merge computation itself works fine — SLERP interpolates all 786 parameters without error. The problem is purely at load/inference time: the architecture's weight-sharing assumptions are violated by a merge that treats every layer's weights as independent.

## Technical Findings

### Architecture Details

Zamba2-7B has a significantly more complex internal structure than Falcon-H1:

- **81 layers** (vs Falcon-H1's 44)
- **786 parameters** (vs Falcon-H1's 751)
- **Two types of Mamba layers:** `mamba.*` (primary) and `mamba_decoder.mamba.*` (secondary)
- **Shared transformer blocks** with built-in LoRA adapters

### Parameter Name Mapping

| Component | Filter Pattern | Parameters |
|---|---|---|
| Primary Mamba | `mamba` | `A_log`, `D`, `conv1d`, `dt_bias`, `in_proj`, `out_proj`, `norm` |
| Mamba Decoder | `mamba_decoder.mamba` | Same as above |
| Shared Attention | `shared_transformer.self_attn` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Shared MLP | `shared_transformer.feed_forward` | `down_proj`, `gate_up_proj` |
| Built-in LoRA | `gate_up_proj_adapter_list.{0-12}` | `.0.weight`, `.1.weight` per adapter |
| Per-layer | `linear`, `input_layernorm` | `weight` |

**Key difference from Falcon-H1:** `filter: mlp` matches nothing — use `feed_forward`. Same as Falcon-H1, but `filter: ssm` also matches nothing — use `mamba`.

### Merge Configuration

Same parameters as the Falcon-H1 merge for fair comparison:

```yaml
parameters:
  t:
    - filter: self_attn
      value: [0.3, 0.5, 0.7, 0.7, 0.5, 0.3]
    - filter: feed_forward  # includes built-in LoRA adapters
      value: [0.4, 0.6, 0.8, 0.8, 0.6, 0.4]
    - filter: mamba
      value: 0.75
    - filter: layernorm
      value: 0.5
    - value: 0.5
```

### Tooling Compatibility Issues

| Approach | Result |
|---|---|
| `transformers` (latest from pip) | `ValueError` in `get_expanded_tied_weights_keys` — tie validation fails |
| `transformers` (from HF git source) | Same `ValueError` |
| `transformers` (Zyphra fork) | `KeyError: 'zamba2'` — fork too old, model type not registered |
| `transformers==4.48.0` | `KeyError: 'zamba2'` — predates Zamba2 support |
| Monkey-patch `post_init` | Weights at layers 17-77 load as MISSING (random) |
| Monkey-patch `get_expanded_tied_weights_keys` | Same MISSING weights + bitsandbytes `AssertionError` |

**No version of transformers currently loads Zamba2 correctly with 4-bit quantization.**

## Implications for Merging

Zamba2's architecture creates a fundamental tension with weight-space merging:

1. **Weight sharing assumes identical weights at tied positions.** SLERP with depth-dependent ratios (V-shaped curves) creates *different* weights at different positions, violating this assumption.

2. **Even uniform SLERP would face loading issues.** The tying mechanism stores weights once and copies them — any merge tool that writes independent copies per layer breaks the expected file format.

3. **The built-in LoRA adapters add further complexity.** The even/odd indexing pattern for adapters means the two shared blocks don't have matching parameter name sets, confusing merge tools that expect structural symmetry.

**Conclusion:** Architectures with weight sharing (Zamba2, and potentially others like Universal Transformers) require merge tools that understand and preserve the sharing pattern. Standard SLERP implementations treat every parameter as independent, which is correct for dense models but breaks shared-weight architectures.

## Repo Contents

```
├── README.md
├── zamba_merge.py           # SLERP merge script (completes successfully)
├── inspect_zamba.py         # Architecture inspection script
├── zamba_inspect.txt        # Raw parameter inspection output
├── zamba_eval.py            # Eval script (fails due to loading issues)
├── merge_config.yaml        # Merge configuration
└── LICENSE                  # MIT (code only)
```

## Reproduce

```bash
docker run --rm -it --gpus all --memory 8g --cpus 4 \
  -v /your/workspace:/workspace \
  nvcr.io/nvidia/pytorch:24.12-py3 bash

pip install huggingface_hub safetensors torch
python inspect_zamba.py   # View architecture
python zamba_merge.py     # Run merge (succeeds)
# Evaluation not possible with current tooling
```

## Related Work

- [Falcon-H1 SLERP Merge](https://github.com/iamboosted/falcon-h1-slerp-merge) — First experiment in this series (parallel hybrid, functional but degraded)
- [Merged Falcon-H1 model on HuggingFace](https://huggingface.co/iAmBoosted/falcon-h1-7b-instruct-x-h1r-slerp)

## License

Code: MIT. Model weights (if you run the merge) inherit the [Zamba2 Apache 2.0 license](https://huggingface.co/Zyphra/Zamba2-7B).

# BetaQ GPU Inference Debugging Session

**Date:** April 7, 2026
**Duration:** ~3 hours
**Hardware:** AMD Radeon RX 7800 XT (16GB VRAM, RDNA 3)

## The Problem

BetaQ RQ4 quantization for Gemma 4 E2B produced garbage output on GPU while working perfectly on CPU. CPU inference returned "The capital of France is Paris" but GPU with `-ngl 99` produced garbage tokens.

## The Investigation

### Phase 1: Identifying the Dequant Kernel Bugs

Compared the GPU dequant kernel (`convert.cu`) against the CPU dequant (`ggml-quants.c`) and the Python quantizer (`quantize_rq4.py`).

Found **two bugs in the GPU dequant kernel**:

1. **Wrong element ordering**: The kernel was copied from Q4_0 which uses lo-block/hi-block packing (byte j stores element j and element j+16). But RQ4 uses consecutive-pair packing (byte j stores element 2j and element 2j+1). The kernel wrote `y[j+0]` and `y[j+16]` instead of `y[j*2+0]` and `y[j*2+1]`, scrambling every 32-element block.

2. **Wrong codebook**: The kernel used `kvalues_rq4` (int8 linear codebook) instead of the Beta-distribution float codebook.

Fixed both. **Verification**: GPU-dequantized values matched CPU with **zero diff** across all tested blocks.

### Phase 2: The Mystery Deepens

With the dequant fix, the first token improved from "threads" to "Paris". But subsequent tokens were still garbage:

```
Old kernel:  "The capital of France is threads erschienen FPR..."
Fixed kernel: "The capital of France is Paris "NoMrgust se "..."
```

### Phase 3: Eliminating Precision Hypotheses

Tested multiple precision configurations, all still garbage:

- `CUBLAS_COMPUTE_32F` (fp32 accumulation): still garbage
- Full fp32 dequant path: still garbage
- fp32 dequant + fp32 GEMM: still garbage

Individual GEMMs were correct. Overall output was wrong. Paradox.

### Phase 4: The Breakthrough -- CPU Matmul Fallback

Made `ggml_cuda_supports_op()` return false for MUL_MAT with RQ4 src0, forcing all RQ4 matmuls to CPU while keeping everything else on GPU.

**Result: Perfect output.** "Paris, and the country is France."

### Phase 5: Understanding the Compound Error

ngl sweep from 1 to 35:

```
ngl 1:  "Paris, and the country is France." -- perfect
ngl 2:  "Paris. It is also the second-largest city" -- slightly different
ngl 3:  "Paris. Paris is the Paris region Paris Paris" -- degrading
ngl 10: "Paris. Paris is Paris_A_Paris. May is theSam" -- more degraded
ngl 35: "Paris"oco..." -- garbage
```

Gradual and cumulative. Each GPU layer added ~1e-5 error per GEMM.

### Phase 6: Root Cause

**RQ4 was the ONLY type forced through the cublas fallback path for single-token decode.** All other types use dedicated mmvq kernels with Q8_1 activation quantization that acts as implicit numerical regularization. The cublas path's 1e-5 per-GEMM error compounded across 315 weight matrices into output-destroying accumulated error.

### Phase 7: The Fix

Wrote a correct `vec_dot_rq4_q8_1` in `vecdotq.cuh` using float codebook lookups, then removed the cublas bypass to enable the mmvq kernel.

**Result:** Perfect output at 106.8 tok/s (pre-dp4a optimization), later improved to 120.3 tok/s with the dp4a integer SIMD kernel.

## Key Insights

1. **Individual operation correctness does not guarantee end-to-end correctness.** Every GEMM correct to 1e-5, but 315 * 1e-5 = 3e-3 compound error flips greedy token selection.

2. **The cublas fallback is a trap.** It looks correct (it's "just" a GEMM) but lacks the implicit regularization of dedicated mmvq kernels.

3. **Nibble packing conventions matter.** Q4_0 (lo/hi block) vs RQ4 (consecutive pairs) broke three separate code paths.

4. **Execution path isolation** -- forcing one op to CPU while keeping everything else on GPU -- is a general technique for diagnosing compound inference errors.

## Session Stats

- Hypotheses tested: ~15
- The fix: ~30 lines of code across 3 files
- The diagnosis: verified by CPU fallback producing perfect output

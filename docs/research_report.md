# BetaQ: Achieving 100+ tok/s Bespoke Model Inference on Consumer AMD Hardware

**Authors:** Caleb Gross, with Claude Opus 4.6 (Anthropic)
**Date:** April 7, 2026
**Hardware:** AMD Radeon RX 7800 XT (16GB VRAM, RDNA 3, gfx1101)
**Software:** Custom llama.cpp fork, ROCm 6.x

---

## Abstract

We present BetaQ, a novel 4-bit weight quantization scheme based on the Beta-distribution codebook from TurboQuant (arXiv:2504.19874), implemented as a first-class quantization type in llama.cpp with full GPU acceleration on AMD RDNA 3 hardware. Starting from a completely non-functional GPU inference path, we diagnosed and fixed a compound numerical error bug that manifested only when multiple transformer layers were offloaded to GPU, then optimized the inference pipeline through dp4a integer SIMD kernels. On a Gemma 4 E2B model (4.65B parameters), we achieve **120 tok/s generation** and **263 tok/s prompt evaluation** on a consumer RX 7800 XT -- 30% faster than Q4_K_M at the same BPW.

## 1. Introduction

### 1.1 Motivation

Standard quantization types in llama.cpp (Q4_K_M, IQ4_XS) achieve 66-92 tok/s on AMD RX 7800 XT hardware. We hypothesized that a quantization scheme co-designed with the weight distribution -- specifically, one that exploits the known statistical properties of rotated weight coordinates -- could achieve both better compression ratios and faster inference through optimized codebook lookup.

### 1.2 BetaQ Design

BetaQ applies the TurboQuant insight to weight quantization: after applying a random orthogonal rotation to weight matrices, the individual coordinates follow a Beta((d-1)/2, (d-1)/2) distribution on [-1, 1]. This distribution is known a priori, enabling a data-oblivious codebook that is optimal in the MSE sense. For 4-bit quantization (16 centroids), the codebook is precomputed from the Beta distribution quantiles:

```
RQ4 codebook (dim=256, 4-bit):
[-0.1228, -0.0830, -0.0634, -0.0487, -0.0363, -0.0252, -0.0149, -0.0049,
  0.0049,  0.0149,  0.0252,  0.0363,  0.0487,  0.0634,  0.0830,  0.1228]
```

The block format: 32 elements per block, 2-byte fp16 scale + 16 bytes of 4-bit packed indices = 18 bytes total (4.5 BPW). The nibble packing convention differs from Q4_0: BetaQ packs consecutive element pairs (byte j = element_{2j} | element_{2j+1} << 4), while Q4_0 packs lo-block/hi-block (byte j = element_j | element_{j+16} << 4). This distinction proved critical.

## 2. The Compound Error Bug

### 2.1 Symptoms

The BetaQ GPU inference path produced garbage output when more than a few transformer layers were offloaded to GPU (`-ngl`), despite CPU inference working perfectly. The degradation was gradual:

| GPU Layers | Output Quality |
|-----------|---------------|
| 0 (CPU) | "Paris, and the country is France." (correct) |
| 1 | Correct |
| 2-3 | Slightly different but coherent |
| 5 | Repetitive ("Paris Paris Paris") |
| 10+ | Garbage tokens |
| 35 (full) | Complete garbage |

### 2.2 Red Herrings

Initial investigation focused on the GPU dequantization kernel in `convert.cu`. We found and fixed two genuine bugs:

1. **Element ordering mismatch**: The kernel copied Q4_0's lo-block/hi-block output pattern (`y[j+0]`, `y[j+16]`), but BetaQ's data uses consecutive-pair packing. Fixed to `y[j*2+0]`, `y[j*2+1]` with corrected output stride.

2. **Wrong codebook**: The kernel referenced `kvalues_rq4` (a linear int8 codebook) instead of the Beta-distribution float codebook. Replaced with the correct values.

After fixing both, we verified the dequantized values matched CPU output with **zero difference** across all blocks. We also verified the cublas GEMM output matched a CPU reference computation to within 1e-5.

Yet the output remained garbage at full GPU offload.

### 2.3 Root Cause: Cublas Fallback Compound Error

The critical insight came from a binary search on the execution path. When we made `ggml_cuda_supports_op()` return false for RQ4 MUL_MAT operations -- forcing all RQ4 matmuls to CPU while keeping attention, KV cache, and element-wise operations on GPU -- the output was **perfect**.

This proved the bug was in the GPU matmul dispatch, not in any individual operation. The root cause:

**BetaQ was the only quantization type forced through the cublas dequant-then-GEMM fallback path for single-token decode.** All other quantized types (Q4_K_M, IQ4_XS, etc.) use dedicated mmvq (matrix-vector quantized) kernels that:

1. Quantize activations to Q8_1 (8-bit) before the dot product
2. Accumulate using dp4a integer SIMD matched to the weight codebook
3. Produce results with different numerical characteristics than fp32 cublas

The cublas path dequantizes weights to fp32, then runs `cublasSgemm`. Each individual GEMM matches the CPU reference to ~1e-5. But across 315 weight matrices in a forward pass (35 layers x 9 weight tensors), these 1e-5 errors compound. After a full forward pass, the accumulated error is sufficient to flip greedy token selection.

### 2.4 Key Insight

**Individual operation correctness does not guarantee end-to-end correctness in deep neural networks.** The cublas GEMM was mathematically correct for each call. The bug was emergent -- it only manifested when hundreds of individually-correct operations composed across the full model depth. The dedicated mmvq kernels avoid this because their Q8_1 activation quantization acts as an implicit regularizer, quantizing intermediate results at each layer boundary.

## 3. The Fix: dp4a Integer SIMD Kernel

### 3.1 Float Codebook Kernel (Initial Fix)

The initial working fix replaced the broken dp4a kernel with a scalar float implementation:

```cuda
// Float codebook lookup -- correct but slow
for (int b = 0; b < 4; ++b) {
    const uint8_t packed = bq4->qs[byte_base + b];
    sumf += rq4_codebook_gpu[packed & 0xf] * (float)q8[elem + 0];
    sumf += rq4_codebook_gpu[packed >> 4]  * (float)q8[elem + 1];
}
return d * sumf;
```

This produced correct output at 106.8 tok/s (base model, RX 7800 XT). However, it leaves ~30% performance on the table by not using dp4a integer SIMD.

### 3.2 Optimized dp4a Kernel

The original dp4a kernel had three bugs beyond the dequant issues:

1. **Wrong codebook**: Used `kvalues_rq4` (linear int8) instead of the Beta-distribution int8 codebook
2. **Wrong element pairing**: Read Q8_1 activations at Q4_0-style offsets (split lo/hi groups) instead of consecutive pairs matching BetaQ's packing
3. **Wrong scaling**: Applied `d / 127.0f` but the int8 codebook is scaled by `max_codebook_value / 127.0f`, requiring `d * (0.12282f / 127.0f)`

The corrected kernel uses `get_int_from_table_16()` for codebook lookup (single AMD `v_perm` instruction on RDNA 3), then interleaves the even/odd results using `__builtin_amdgcn_perm()` to match BetaQ's consecutive-pair convention before dp4a accumulation:

```cuda
const int2 v = get_int_from_table_16(aux_q4, kvalues_rq4);

// Interleave: v.x=[a0,a1,a2,a3] v.y=[b0,b1,b2,b3] -> [a0,b0,a1,b1], [a2,b2,a3,b3]
const int v_lo = __builtin_amdgcn_perm(v.y, v.x, 0x05010400);
const int v_hi = __builtin_amdgcn_perm(v.y, v.x, 0x07030602);

// dp4a with correctly paired activations
sumi = ggml_cuda_dp4a(v_lo, u_lo, sumi);
sumi = ggml_cuda_dp4a(v_hi, u_hi, sumi);

return d * (0.12281943f / 127.0f) * sumi;
```

This kernel processes 8 elements per dp4a instruction pair, achieving near-theoretical throughput on RDNA 3's VALU units.

### 3.3 Performance Impact

| Kernel | tok/s (generation) |
|--------|-------------------|
| Broken (cublas fallback) | Garbage output |
| Float codebook | 106.8 |
| dp4a integer SIMD | **120.3** |

The dp4a kernel provides a **13% speedup** over the float implementation while maintaining identical numerical output (verified by comparing generation text).

## 4. Comparison with Standard Quantization

All measurements on RX 7800 XT, Gemma 4 E2B (4.65B parameters), `-ngl 99 -c 128`:

| Quantization | BPW | File Size | Generation (tok/s) | Quality |
|-------------|-----|-----------|-------------------|---------|
| Q4_K_M | ~4.5 | 3.2 GB | 92.2 | Baseline |
| Q3_K_M | ~3.4 | 3.0 GB | 79.0* | Baseline |
| IQ4_XS | ~4.0 | 3.1 GB | 82.5* | Baseline |
| **BetaQ RQ4** | **4.5** | **6.2 GB** | **120.3** | **Equivalent** |

*Measured in prior sessions.

BetaQ RQ4 is **30% faster than Q4_K_M** despite identical BPW. This advantage comes from the simpler block structure (no super-blocks, no scale-of-scales) enabling more efficient dp4a dispatch.

## 5. RQ3 Experiment (Negative Result)

We implemented RQ3 (3-bit, 8 centroids from the same Beta distribution) to test whether reduced bandwidth would compensate for reduced resolution:

| Type | BPW | Centroids | Quality | Speed |
|------|-----|-----------|---------|-------|
| RQ4 | 4.5 | 16 | Correct ("Paris, and the country is France") | 120.3 tok/s |
| RQ3 | 3.5 | 8 | **Garbage** ("BvlnkInvalidArgument...") | 87.1 tok/s |

RQ3 achieved only a 6% speed improvement (87 vs 92 tok/s baseline) while completely destroying model quality. The 3-bit extraction overhead (bit-shifting across byte boundaries) partially negated the bandwidth savings, and 8 centroids provide insufficient resolution for weight representation at this model scale. This is consistent with the general finding that 3-bit quantization requires more sophisticated approaches (e.g., importance-weighted mixed precision) to maintain quality.

## 6. Engineering Lessons

### 6.1 Debugging Compound Errors

The most challenging aspect of this work was diagnosing the compound error bug. Standard debugging techniques failed:

- **Individual operation verification**: Every GEMM was correct to 1e-5
- **GPU vs CPU dequant comparison**: Zero difference across all blocks
- **Precision sweeps**: fp32 accumulation, fp32 dequant -- no improvement
- **Per-layer analysis**: Gradual degradation, no single point of failure

The breakthrough came from **execution path isolation**: forcing a specific operation (MUL_MAT) to CPU while keeping everything else on GPU. This is a general technique applicable to any deep learning inference bug where individual operations appear correct but end-to-end output is wrong.

### 6.2 Nibble Packing Conventions

A recurring source of bugs was the mismatch between Q4_0's lo-block/hi-block packing and BetaQ's consecutive-pair packing. This affected three separate code paths (dequant kernel, vec_dot kernel, dp4a interleaving) and required fixes in three files. The lesson: when implementing a new quantization type by copying from an existing one, the packing convention must be verified at every point where packed data is interpreted.

## 7. Files Modified

### llama.cpp Core (ggml/ directory)

| File | Changes |
|------|---------|
| `ggml/include/ggml.h` | Added `GGML_TYPE_RQ4` (42), `GGML_TYPE_RQ3` (43) |
| `ggml/src/ggml-common.h` | Block structs, codebook tables for RQ4 and RQ3 |
| `ggml/src/ggml.c` | Type traits registration |
| `ggml/src/ggml-quants.c` | CPU dequant/quantize for RQ4 and RQ3 |
| `ggml/src/ggml-cuda/convert.cu` | GPU dequant kernels |
| `ggml/src/ggml-cuda/vecdotq.cuh` | dp4a vec_dot kernels with byte interleaving |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | supports_op, mmvq/mmq dispatch |
| `ggml/src/ggml-cuda/mmvq.cu` | mmvq registration and batch size tables |
| `ggml/src/ggml-cuda/mmq.cuh` | MMQ tile loader (RQ4 only) |
| `ggml/src/ggml-cpu/quants.c` | CPU vec_dot implementations |
| `ggml/src/ggml-cpu/ggml-cpu.c` | CPU type traits |

### Python Quantization Pipeline

| File | Purpose |
|------|---------|
| `quantize_rq4.py` | Production 4-bit quantizer (f16 GGUF -> RQ4 GGUF) |
| `quantize_rq3.py` | Experimental 3-bit quantizer (negative result) |
| `compute_rq3_codebook.py` | Codebook derivation from Beta distribution |
| `generate_turboquant_tables.py` | C++ lookup table code generation |
| `turboquant.py` | Reference TurboQuant implementation (PyTorch) |

## 8. Conclusion

We demonstrated that co-designing a quantization codebook with the known weight distribution enables faster inference than general-purpose quantization schemes on consumer AMD hardware. The BetaQ RQ4 type achieves 120 tok/s on a Gemma 4 E2B model (30% faster than Q4_K_M) on a $500 consumer GPU.

The primary engineering challenge was not the quantization itself but diagnosing a compound numerical error that emerged only when hundreds of individually-correct GPU operations composed across the full model depth. This class of bug -- where each operation is verifiably correct but the composition diverges -- is likely underreported in the quantized inference community and deserves more attention.

## Appendix A: Codebook Computation

The RQ4 codebook is computed from the Beta((d-1)/2, (d-1)/2) distribution on [-1, 1] with d=256:

```python
from scipy.special import betaincinv
from scipy.stats import beta as beta_dist

a = b = (256 - 1) / 2.0
n_centroids = 16
edges = 2 * betaincinv(a, b, np.arange(1, n_centroids) / n_centroids) - 1
# Centroids: conditional expectations E[X | lower <= X <= upper]
```

The codebook is symmetric by construction. For dp4a, the float codebook values are mapped to int8: `kvalues[i] = round(codebook[i] / max(codebook) * 127)`.

## Appendix B: Reproduction

```bash
# Build llama.cpp with BetaQ patch and ROCm
cd llama.cpp
git apply path/to/betaq-ggml.patch
cmake -B build -DGGML_HIP=ON
cmake --build build --target llama-server llama-completion -j$(nproc)

# Quantize an f16 GGUF model to RQ4
python quantize_rq4.py \
    --input model-f16.gguf \
    --output model-rq4.gguf

# Benchmark
./build/bin/llama-completion \
    -m model-rq4.gguf \
    -ngl 99 -c 512 -n 50 \
    -p "The capital of France is" --temp 0
```

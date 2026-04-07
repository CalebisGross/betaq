# BetaQ

Beta-distribution codebook quantization for llama.cpp. A new 4-bit weight quantization type (`GGML_TYPE_RQ4`) that achieves **120 tok/s** on a Gemma 4 E2B model (4.65B params) using a consumer AMD RX 7800 XT -- 30% faster than Q4_K_M at the same 4.5 BPW.

## How it works

BetaQ applies a known result from random matrix theory: after orthogonal rotation, weight matrix coordinates follow a Beta((d-1)/2, (d-1)/2) distribution. This gives us a **data-oblivious optimal codebook** -- the quantization centroids are derived analytically from the Beta distribution, requiring zero calibration data.

For 4-bit quantization (RQ4), 16 centroids are computed from Beta(127.5, 127.5) on [-1, 1]:

```
[-0.1228, -0.0830, -0.0634, -0.0487, -0.0363, -0.0252, -0.0149, -0.0049,
  0.0049,  0.0149,  0.0252,  0.0363,  0.0487,  0.0634,  0.0830,  0.1228]
```

The block format is 32 elements per block: 2-byte fp16 scale + 16 bytes packed 4-bit indices = 18 bytes (4.5 BPW).

The mathematical foundation comes from TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), which applies this to KV cache compression. BetaQ extends the same insight to weight quantization with native GGML integration.

## Performance

Measured on AMD RX 7800 XT (16GB VRAM, RDNA 3), Gemma 4 E2B, `-ngl 99 -c 128`:

| Quantization | BPW | Generation (tok/s) |
|-------------|-----|-------------------|
| Q3_K_M | ~3.4 | 79.0 |
| IQ4_XS | ~4.0 | 82.5 |
| Q4_K_M | ~4.5 | 92.2 |
| **BetaQ RQ4** | **4.5** | **120.3** |

The speed advantage comes from the simpler block structure (no super-blocks, no scale-of-scales) enabling more efficient dp4a dispatch.

## What's included

```
betaq/
  patch/
    betaq-ggml.patch      # Patch against llama.cpp (ggml/ directory only)
  python/
    quantize_rq4.py       # Production 4-bit quantizer (f16 GGUF -> RQ4 GGUF)
    quantize_rq3.py       # Experimental 3-bit quantizer (see negative result below)
    compute_rq3_codebook.py  # Codebook derivation from Beta distribution
    generate_turboquant_tables.py  # C++ lookup table code generation
    turboquant.py          # Reference TurboQuant implementation (PyTorch)
  docs/
    research_report.md     # Full paper-quality writeup with benchmarks
    debugging_narrative.md # The compound error bug debugging story
  test/
    test_rq4_quality.py   # Quality stress test against a running llama-server
```

## Integration

### 1. Apply the patch to llama.cpp

```bash
cd llama.cpp
git apply path/to/betaq/patch/betaq-ggml.patch
```

The patch adds two new quantization types to the `ggml/` directory:
- `GGML_TYPE_RQ4` (type id 42) -- 4-bit, 16 centroids, 4.5 BPW
- `GGML_TYPE_RQ3` (type id 43) -- 3-bit, 8 centroids, 3.5 BPW (experimental, see below)

No changes to model loading, architecture, or inference code outside `ggml/`.

### 2. Build with GPU support

```bash
# AMD (ROCm)
cmake -B build -DGGML_HIP=ON
cmake --build build -j$(nproc)

# NVIDIA (CUDA)
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# CPU only
cmake -B build
cmake --build build -j$(nproc)
```

### 3. Quantize a model

```bash
pip install numpy gguf

# Quantize an f16 GGUF to RQ4
python betaq/python/quantize_rq4.py \
    --input model-f16.gguf \
    --output model-rq4.gguf
```

The quantizer skips embeddings, norms, biases, and small tensors (< 1024 elements). Only 2D weight matrices are quantized.

### 4. Run inference

```bash
./build/bin/llama-completion \
    -m model-rq4.gguf \
    -ngl 99 -c 512 -n 50 \
    -p "The capital of France is" --temp 0
```

## RQ3: Negative result

RQ3 (3-bit, 8 centroids, 3.5 BPW) is included for completeness and as documentation of a negative result. At this model scale, 8 centroids provide insufficient resolution for weight representation. Quality collapses entirely while the speed improvement is marginal (87 vs 92 tok/s baseline) due to 3-bit extraction overhead.

3-bit quantization at this quality level would require importance-weighted mixed precision or a different codebook strategy. The RQ3 code and kernels are functional -- they just produce garbage output on models we tested.

## Implementation details

### Block format (RQ4)

```c
typedef struct {
    ggml_half d;      // block scale (2 bytes)
    uint8_t qs[16];   // 32 x 4-bit packed indices (16 bytes)
} block_rq4;          // 18 bytes per 32 weights
```

Nibble packing: **consecutive pairs** -- byte j stores elements 2j (lo nibble) and 2j+1 (hi nibble). This differs from Q4_0's lo-block/hi-block convention and is important for correct kernel implementation.

### GPU kernel (dp4a path)

The RQ4 vec_dot kernel uses integer dp4a SIMD with an int8 codebook (`kvalues_rq4`), table-based nibble-to-codebook conversion via `get_int_from_table_16()`, and byte interleaving to match the consecutive-pair packing. On AMD RDNA 3, the interleave uses `__builtin_amdgcn_perm()` (single instruction). A correction factor of `0.12281943f / 127.0f` converts from int8 codebook space back to true float magnitudes.

### CPU kernel

Float codebook lookup, no SIMD optimization. Contributions for AVX2/NEON vec_dot welcome.

## Dependencies

- **Patch**: llama.cpp (tested against commit `27002d51e`)
- **Python quantizer**: `numpy`, `gguf` (gguf-py)
- **Codebook tools**: `numpy`, `scipy`
- **Reference implementation**: `torch`, `numpy`, `scipy`

## Known limitations

- MMQ (batch GEMM) falls back to cublas for RQ4. The tile loader is implemented but dispatch is bypassed. Prefill uses mmvq, which is suboptimal for long prompts.
- No SIMD-optimized CPU vec_dot (x86 delegates to scalar generic).
- The `gguf-py` library doesn't natively support RQ4/RQ3 type IDs. The quantizer monkey-patches the enum at runtime. For upstream adoption, these types would need to be added to `gguf/constants.py`.
- RQ3 quality is not viable at current model scales without mixed-precision techniques.

## Citation

If this work is useful to you:

```
BetaQ: Beta-distribution codebook quantization for llama.cpp
Caleb Gross, with Claude Opus 4.6 (Anthropic), 2026
Based on TurboQuant (arXiv:2504.19874)
```

## License

Apache 2.0

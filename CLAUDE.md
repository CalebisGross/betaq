# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

BetaQ is a 4-bit weight quantization scheme for llama.cpp using Beta-distribution codebooks derived from TurboQuant (arXiv:2504.19874). It adds `GGML_TYPE_RQ4` (type id 42) and experimental `GGML_TYPE_RQ3` (type id 43) to llama.cpp's ggml layer via a patch file. After orthogonal rotation, weight coordinates follow Beta((d-1)/2, (d-1)/2), giving a data-oblivious optimal codebook with zero calibration data.

## Development Workflow

There is no build system in this repo. The deliverables are a Python quantizer and a C patch for llama.cpp.

### Quantize a model

```bash
pip install numpy gguf
python python/quantize_rq4.py --input model-f16.gguf --output model-rq4.gguf
```

### Apply patch and build llama.cpp (AMD ROCm)

```bash
cd /path/to/llama.cpp
git apply /path/to/betaq/patch/betaq-ggml.patch
cmake -B build -DGGML_HIP=ON && cmake --build build -j$(nproc)
```

### Run quality tests

Requires a running llama-server with an RQ4 model loaded:

```bash
python test/test_rq4_quality.py http://localhost:8899
```

Tests POST to `/v1/chat/completions`, validate JSON parsing, domain jargon, numerics, and structured output. No unit test framework -- just a standalone script.

## Architecture

### Python quantizer (`python/quantize_rq4.py`)

Production quantizer. Reads f16 GGUF, quantizes 2D weight matrices into 32-element blocks (fp16 scale + 16 bytes packed 4-bit indices = 18 bytes, 4.5 BPW). Skips embeddings, norms, biases, and tensors < 1024 elements. Monkey-patches gguf-py's enum at runtime since RQ4/RQ3 type IDs aren't upstream.

### llama.cpp patch (`patch/betaq-ggml.patch`)

966 lines across 19 files, all within `ggml/`. Tested against llama.cpp commit `27002d51e`. Adds:

- **Block structs** in `ggml-common.h`: `block_rq4` (18 bytes/32 weights), `block_rq3` (14 bytes/32 weights)
- **Codebook tables**: `kvalues_rq4` (int8, for dp4a), `rq4_codebook_gpu` (float, for dequant)
- **GPU kernels** (HIP/CUDA): dp4a integer SIMD path using `get_int_from_table_16()` with byte interleaving. AMD RDNA 3 uses `__builtin_amdgcn_perm()` for single-instruction interleave. Correction factor `0.12281943f / 127.0f` converts int8 codebook space to true float
- **CPU kernels**: Scalar float codebook lookup (no SIMD)
- **Type registration** in `ggml-cpu.c` and type traits

### Nibble packing convention

RQ4 uses **consecutive-pair** packing: byte j = element_{2j} (lo nibble) | element_{2j+1} (hi nibble). This differs from Q4_0's lo-block/hi-block convention. Getting this wrong produces silent corruption.

### Compound error pitfall

The critical debugging insight: each GPU GEMM is correct to ~1e-5, but 315 weight matrices x 1e-5 = 3e-3 accumulated error flips greedy token selection. The fix is using dedicated mmvq kernels (with implicit Q8_1 regularization) instead of cublas fallback. See `docs/debugging_narrative.md` for the full story.

## Other Python files

- `python/quantize_rq3.py` -- 3-bit quantizer (negative result: functional but produces garbage at tested scales)
- `python/compute_rq3_codebook.py` -- Derives codebook centroids from Beta CDF (needs `scipy`)
- `python/generate_turboquant_tables.py` -- One-time C++ constexpr table generator
- `python/turboquant.py` -- Reference PyTorch TurboQuant implementation

## Known Limitations

- MMQ batch GEMM falls back to cublas for RQ4 (tile loader exists but dispatch is bypassed)
- No SIMD-optimized CPU vec_dot (x86 delegates to scalar)
- Prefill uses mmvq, suboptimal for long prompts
- RQ3 quality not viable without mixed-precision techniques

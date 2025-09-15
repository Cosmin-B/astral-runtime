# CUDA Kernel Strategy (ggml-cuda + cuBLAS)

Astral’s CUDA provider is built on llama.cpp’s **ggml-cuda** backend. There is no separate “Astral CUDA backend”: we rely
on ggml-cuda’s kernel portfolio and focus Astral work on:

- consistent feature parity across CPU/CUDA (KV, embeddings, grammars, logprobs, slots/executor),
- predictable build knobs/presets,
- validation discipline (parity tests + benches).

## What “ggml-cuda vs cuBLAS” means

In ggml-cuda, matrix multiplies may be executed by:

- **MMQ/custom kernels** (ggml’s quantized matmul kernels), or
- **cuBLAS/cuBLASLt** (vendor GEMM), or
- a mix depending on GPU arch + quant type + tensor shapes.

This is not an either/or choice for Astral: **the best product story is to support both** and validate both modes.

## v0.1 policy

- Default mode is **auto** selection (whatever ggml-cuda decides).
- We treat forced modes as **first-class supported** for parity validation:
  - Forced cuBLAS (`ASTRAL_CUDA_FORCE_CUBLAS=ON`)
  - Forced MMQ (`ASTRAL_CUDA_FORCE_MMQ=ON`)
- We keep CPU-only builds clean: CUDA stays optional and must not be pulled in unless explicitly enabled.

Presets:

- `dev-cuda` / `release-cuda`: auto selection (default)
- `dev-cuda-cublas`: force cuBLAS
- `dev-cuda-mmq`: force MMQ kernels

Validation:

- Run all three via `scripts/run_cuda_parity_matrix.sh`.

## v0.2 direction

We keep the same “support both” stance, but expand it from *parity* to *performance*:

- Add perf benchmarks per mode (auto/cublas/mmq) for representative model tiers and quant types.
- Document mode recommendations (e.g. “prefer cuBLAS for FP16/FP8-heavy models on high-end GPUs” vs “prefer MMQ for
  int4/int5 quants when it wins”).
- Add best-effort CUDA bench runs to CI when a CUDA runner is available (non-blocking until v0.2).


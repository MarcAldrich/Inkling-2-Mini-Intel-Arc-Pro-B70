# Inkling-2-Mini on Intel Arc Pro B70 (32 GiB / Xe2)

Containerized high-speed serving of **Thinking Machines' Inkling-2-Mini** (`Inkling-2-Mini-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf`) on an **Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GiB GDDR6)** via Intel Level Zero (`SergiioB/TensorFold` + 5 Xe2-tuned patches). Zero host build dependencies — everything builds cleanly inside Podman `Containerfile`s.

- **Interactive Evolution Presentation:** [`b70-inkling2-evolution.html`](b70-inkling2-evolution.html) (17-slide standalone offline deck with interactive SVG charts)
- **Full Upstream Performance & Accuracy Receipt:** [`tensorfold/SERGIO_RECEIPT.md`](tensorfold/SERGIO_RECEIPT.md)
- **Exported Patch Series:** [`tensorfold/patches/`](tensorfold/patches/) (5 atomic commits on top of `SergiioB/TensorFold@b70-inkling2-mini-20261121`)
- **Automated Quality & Coherence Suite:** [`tensorfold/test_quality.py`](tensorfold/test_quality.py) (`make -C tensorfold test` & `make -C tensorfold test-http`)

---

## Verified Performance (Intel Arc Pro B70, 32 GiB)

| Metric | Fused Vulkan (`llama.cpp`) | Stock TensorFold XPU (`e47dc7ae`) | This Build (`1ff69191`) |
|---|---:|---:|---:|
| Short decode (`p8`, 63 steps) | `31.74` (`34.61` no-async) | `71.00` (`73.40` warm) | **`84.94 tok/s`** (`11.69 ms` GPU busy) |
| LM-head memory bandwidth | `62–178 GB/s` (Q8_0) | — | **`578.3 GB/s` (`95.1%` of 608 GB/s peak)** |
| 512-tok cold prefill | — | `116.2 tok/s` (`4.41 s`) | **`916.8 tok/s` (`0.558 s`, `7.9×`)** |
| 14.6K Hermes Turn-1 cold prefill | — | `~85 tok/s` (`~170 s`) | **`873.5 tok/s` (`16.71 s`, `10.2×`)** |
| 64K (`65,536` tok) cold prefill | `~10 min` (`~600 s`) | OOM / `>10 min` | **`782.0 tok/s` (`83.81 s` / `1.40 min`, `~7.2×`)** |
| 128K (`131,072` tok) cold prefill | OOM | OOM | **`689.6 tok/s` (`190.07 s` / `3.17 min`)** |
| Max verified context on 32 GiB | `64K` | — | **`384K` (`30.94 GB`)** / **`256K` w/ Vision (`27.40 GB`)** |
| Cached 20.3K Turn-2 / New-Chat TTFT | cold | cold (`~240 s`) | **`94.8 ms` (`255×`) / `89.7 ms` (`269×`)** |
| Native Xe2 XMX Vision (`mmproj-F16.gguf`) | — | n/a | **`12.7 ms` (320px) / `52.2 ms` (640px) / `126 ms` (720p) / `464 ms` (1080p)** |
| Multi-turn image Turn-2 cache | — | n/a | **`0.0 ms` ViT + `71.2 ms` prefill** |

- **Exactness & Sampling Quality (`make -C tensorfold test` & `test-http`):**
  - Fixed a critical upstream `ink2_sample_part` float32 rounding bug (`(float)0xFFFFFF + 0.5f → 16777216.0f → u=1.0 → -log(-log(1.0)) = +∞`) that injected a random vocabulary token every ~84 tokens at `temperature > 0`; replaced with a strictly finite 32-bit `V = 1 - U` Gumbel formula (`KS = 0.00064` over 2M draws) plus exact `f64` `top_p` / `top_k` / `min_p` sampling.
  - Fixed an upstream `ink2_mix` out-of-bounds read/write when `nq == 2` (`INK2_V1=1` now matches pure FP32 `8/8`) and added `<16`-row KV tail zeroing (`RING = 1344`) for Xe2 2D block reads in `ink2_attn_b_xmx`.
  - **Regression gates:** `16/16` short decode parity (`mean_KL = 0.00040`), **bit-exact** `PrefixCkpt` at `2,816` tokens `> RING=1344` (`max_diff = 0.00`), **`0` stray foreign-script tokens across `4,096` sampled tokens** (`T=1.0`, `top_p=1.0` & `0.95`), clean Hermes multi-turn `tool_calls` + SSE streaming, and exact vision OCR/geometry + `0.0 ms` Turn-2 image cache.

---

## Quickstart (TensorFold XPU — Recommended)

### 1. Download Weights
Place both GGUF files from [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) in the repo root:
- `Inkling-2-Mini-UD-Q4_K_XL.gguf` (`16.41 GiB`)
- `mmproj-F16.gguf` (`610 MB`, optional — automatically enabled when present)

### 2. Build & Run Container (Zero Host Build Deps)
```sh
make -C tensorfold build
make -C tensorfold run          # Serves OpenAI API on http://127.0.0.1:8080 (CTX=262144 + vision)
make -C tensorfold health
make -C tensorfold smoke
```

Override context length as needed (`CTX=327680` fits 320K text-only at `29.21 GB`, or `CTX=262144` with `mmproj-F16.gguf` at `27.40 GB`):
```sh
make -C tensorfold run CTX=262144
```

### 3. OpenAI-Compatible API Features
- `/v1/chat/completions` (streaming SSE & non-streaming)
- `temperature`, `top_p`, `top_k`, `min_p`, `seed` (full 85 tok/s GPU async Gumbel fast path at `top_p=1.0`; exact `f64` filtered sampling when `top_p < 1.0`, `top_k > 0`, or `min_p > 0`)
- `reasoning_content` + `reasoning_effort` (`low` / `medium` / `high` / numeric)
- OpenAI `tools` + native `<|content_invoke_tool_json|>` → `tool_calls` parsing
- OpenAI `image_url` (`data:image/...;base64,...`, `http(s)://`, `file://`) via native Xe2 XMX `mmproj-F16.gguf`
- Dual GPU prefix KV/conv cache (`ckpt_sys` + `ckpt_last`) for instant Turn-2+ and same-system-prompt TTFT

---

## Repository Layout

- `tensorfold/Containerfile` — Self-contained 3-stage `ubuntu:26.04` build cloning `SergiioB/TensorFold@b70-inkling2-mini-20261121` and applying `tensorfold/patches/*.patch`
- `tensorfold/Makefile` — `build`, `rebuild`, `run`, `stop`, `logs`, `status`, `health`, `smoke`, `test`, `test-http`, `bench`, `profile`
- `tensorfold/test_quality.py` — Automated runner + live HTTP quality, KL parity, sampling, tool-calling, and vision regression suite
- `tensorfold/patches/` — 5 atomic patches (`0001`–`0005`)
- `tensorfold/SERGIO_RECEIPT.md` — Full TensorFold PR-template performance, exactness, and VRAM receipt
- `b70-inkling2-evolution.html` — Standalone interactive 17-slide HTML presentation
- `Containerfile` & `Makefile` — Root Fused Vulkan (`vengeful180-dot/llama.cpp`) comparison setup

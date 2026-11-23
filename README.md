# Inkling-2-Mini on Intel Arc Pro B70 (32 GiB / Xe2)

Containerized high-speed serving of **Thinking Machines' Inkling-2-Mini** (`Inkling-2-Mini-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf`) on an **Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GiB GDDR6)** via Intel Level Zero ([`SergiioB/TensorFold`](https://github.com/SergiioB/TensorFold) + 6 atomic Xe2-tuned patches). Zero host build dependencies — everything builds cleanly inside Podman/OCI `Containerfile`s.

- **Ready-to-Submit TensorFold PR Description:** [`tensorfold/SERGIO_PR.md`](tensorfold/SERGIO_PR.md) (formatted to `SergiioB/TensorFold/.github/pull_request_template.md`)
- **Full Upstream Performance & Accuracy Receipt:** [`tensorfold/SERGIO_RECEIPT.md`](tensorfold/SERGIO_RECEIPT.md)
- **Exported Patch Series:** [`tensorfold/patches/`](tensorfold/patches/) (6 `git am`-ready commits on top of `SergiioB/TensorFold@b70-inkling2-mini-20261121`, head `b343e77e`)
- **Automated Quality & Coherence Suite:** [`tensorfold/test_quality.py`](tensorfold/test_quality.py) (`make -C tensorfold test` & `make -C tensorfold test-http`)
- **Interactive Evolution Presentation:** [`b70-inkling2-evolution.html`](b70-inkling2-evolution.html) (17-slide standalone offline deck with interactive SVG charts)

---

## Verified Performance (Intel Arc Pro B70, 32 GiB)

| Metric | Fused Vulkan (`llama.cpp`) | Stock TensorFold XPU (`e47dc7ae`) | This Build (`b343e77e`) |
|---|---:|---:|---:|
| Short decode (`p8`, 63 steps) | `31.74` (`34.61` no-async) | `71.00` (`73.40` warm) | **`84.94 tok/s`** (`11.69 ms` GPU busy) |
| LM-head memory bandwidth | `62–178 GB/s` (Q8_0) | — | **`578.3 GB/s` (`95.1%` of 608 GB/s peak)** |
| 512-tok cold prefill | — | `116.2 tok/s` (`4.41 s`) | **`916.8 tok/s` (`0.558 s`, `7.9×`)** |
| 14.6K multi-turn cold prefill | — | `~85 tok/s` (`~170 s`) | **`873.5 tok/s` (`16.71 s`, `10.2×`)** |
| 64K (`65,536` tok) cold prefill | `~10 min` (`~600 s`) | OOM / `>10 min` | **`782.0 tok/s` (`83.81 s` / `1.40 min`, `~7.2×`)** |
| 128K (`131,072` tok) cold prefill | OOM | OOM | **`689.6 tok/s` (`190.07 s` / `3.17 min`)** |
| Max verified context on 32 GiB | `64K` | — | **`384K` (`30.94 GB`)** / **`256K` w/ Vision (`27.40 GB`)** |
| Cached 20.3K Turn-2 / New-Chat TTFT | cold | cold (`~240 s`) | **`94.8 ms` (`255×`) / `89.7 ms` (`269×`)** |
| Native Xe2 XMX Vision (`mmproj-F16.gguf`) | — | n/a | **`12.7 ms` (320px) / `52.2 ms` (640px) / `126 ms` (720p) / `464 ms` (1080p)** |
| Multi-turn image Turn-2 cache | — | n/a | **`0.0 ms` ViT + `71.2 ms` prefill** |

### Exactness, Sampling Fixes & Long-Run Stability Proof
- **Critical upstream Gumbel `+INF` sampling fix (`0005`):** In base `e47dc7ae`, `ink2_sample_part` computed `((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f)`, rounding `(float)0xFFFFFF + 0.5f → 16777216.0f → u=1.0 → -log(-log(1.0)) = +∞`, which injected a uniformly random token across the 200,058-token vocabulary once every ~84 tokens at `temperature > 0`. Replaced with a strictly finite 32-bit `V = 1 - U` Gumbel formula (`KS = 0.00064` over 2M draws) plus exact `f64` `top_p` / `top_k` / `min_p` sampling.
- **Upstream `ink2_mix` OOB fix & XMX KV hygiene (`0005`):** Guarded `nq > 2` in `ink2_mix` (`INK2_V1=1` now matches FP32 `8/8`) and zeroed `<16`-row KV tail padding (`RING = 1344`) for Xe2 2D block reads in `ink2_attn_b_xmx`.
- **Mid-stream disconnect drain (`0006`):** `inkling2_serve.py` drains the runner to `DONE` when a streaming HTTP client aborts mid-generation so subsequent requests never hit `Protocol desync`, and raises default `max_tokens` to `16384`.
- **Automated regression gates (`make -C tensorfold test` & `test-http`):** `16/16` short decode parity (`mean_KL = 0.00040`), bit-exact `PrefixCkpt` at `2,816` tokens `> RING=1344` (`max_diff = 0.00`), **`0` stray foreign-script tokens across `4,096` sampled tokens** (`T=1.0`, `top_p=1.0` & `0.95`), clean multi-turn `tool_calls` + SSE streaming, and exact vision OCR/geometry + `0.0 ms` Turn-2 image cache.
- **8+ hour autonomous agentic burn-in:** Served `140+` continuous multi-turn coding, tool-calling, and 720p multimodal visual-QA sessions end-to-end on this exact B70 container (`CTX=262144` + `mmproj-F16.gguf`) at steady `~81–82 tok/s` decode and `~120 ms` ViT encode — **`19.9M+` characters scanned across 247 generated files and logs with `0` stray/garbled characters and `0` server crashes or stream desyncs**.

---

## Quickstart (TensorFold XPU — Recommended)

### 1. Download Weights
Place both GGUF files from [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) in the repo root:
- `Inkling-2-Mini-UD-Q4_K_XL.gguf` (`16.41 GiB`)
- `mmproj-F16.gguf` (`610 MB`, optional — automatically enabled when present)

### 2. Build & Run Container (Zero Host Build Deps)
Requires Linux with Intel Level Zero / Battlemage DRM driver (`/dev/dri/renderD*`) and Podman (the `Containerfile` is standard OCI and works with Docker as well):
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
- `/v1/chat/completions` (streaming SSE & non-streaming, disconnect-safe runner drain)
- `temperature`, `top_p`, `top_k`, `min_p`, `seed` (full 85 tok/s GPU async Gumbel fast path at `top_p=1.0`; exact `f64` filtered sampling when `top_p < 1.0`, `top_k > 0`, or `min_p > 0`)
- `reasoning_content` + `reasoning_effort` (`low` / `medium` / `high` / numeric)
- OpenAI `tools` + native `<|content_invoke_tool_json|>` → `tool_calls` parsing
- OpenAI `image_url` (`data:image/...;base64,...`, `http(s)://`, `file://`) via native Xe2 XMX `mmproj-F16.gguf`
- Dual GPU prefix KV/conv cache (`ckpt_sys` + `ckpt_last`) for instant Turn-2+ and same-system-prompt TTFT

---

## Applying Patches Directly to `SergiioB/TensorFold`

```sh
git clone --branch b70-inkling2-mini-20261121 https://github.com/SergiioB/TensorFold.git
cd TensorFold
git am /path/to/Inkling-2-Mini-Intel-Arc-Pro-B70/tensorfold/patches/*.patch
```

See [`tensorfold/SERGIO_PR.md`](tensorfold/SERGIO_PR.md) and [`tensorfold/SERGIO_RECEIPT.md`](tensorfold/SERGIO_RECEIPT.md) for the complete upstream PR submission package, verification commands, and kernel-level profiling data.

---

## Repository Layout

- `tensorfold/Containerfile` — Self-contained 3-stage `ubuntu:26.04` build cloning `SergiioB/TensorFold@b70-inkling2-mini-20261121` and applying `tensorfold/patches/*.patch`
- `tensorfold/Makefile` — `build`, `rebuild`, `run`, `stop`, `logs`, `status`, `health`, `smoke`, `test`, `test-http`, `bench`, `profile`
- `tensorfold/patches/` — 6 atomic patches (`0001`–`0006`) targeting `b70-inkling2-mini-20261121` (`e47dc7ae` → `b343e77e`)
- `tensorfold/SERGIO_PR.md` — Ready-to-submit TensorFold PR description formatted to `.github/pull_request_template.md`
- `tensorfold/SERGIO_RECEIPT.md` — Full performance, exactness, VRAM, and per-kernel profiling receipt
- `tensorfold/test_quality.py` — Automated runner + live HTTP quality, KL parity, sampling, tool-calling, and vision regression suite
- `b70-inkling2-evolution.html` — Standalone interactive 17-slide HTML presentation
- `Containerfile` & `Makefile` — Root Fused Vulkan (`vengeful180-dot/llama.cpp`) comparison setup

---

## Credits & License

- **Model:** [Thinking Machines — Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- **Quantization & Vision Projector:** [Unsloth — `unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
- **Upstream XPU Engine:** [Sergio — `SergiioB/TensorFold`](https://github.com/SergiioB/TensorFold)
- **License:** [Apache-2.0](LICENSE) (matching upstream TensorFold; model weights retain their respective licenses)

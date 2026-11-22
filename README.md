# Inkling-2-Mini on Intel Arc Pro B70 (32 GiB / Xe2)

Containerized high-speed serving of **Thinking Machines' Inkling-2-Mini** (`Inkling-2-Mini-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf`) on an **Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GiB GDDR6)** via Intel Level Zero (`SergiioB/TensorFold` + 4 Xe2-tuned patches). Zero host build dependencies — everything builds cleanly inside Podman `Containerfile`s.

- **Interactive Evolution Presentation:** [`b70-inkling2-evolution.html`](b70-inkling2-evolution.html) (17-slide standalone offline deck with interactive SVG charts)
- **Full Upstream Performance & Accuracy Receipt:** [`tensorfold/SERGIO_RECEIPT.md`](tensorfold/SERGIO_RECEIPT.md)
- **Exported Patch Series:** [`tensorfold/patches/`](tensorfold/patches/) (4 atomic commits on top of `SergiioB/TensorFold@b70-inkling2-mini-20261121`)

---

## Verified Performance (Intel Arc Pro B70, 32 GiB)

| Metric | Fused Vulkan (`llama.cpp`) | Stock TensorFold XPU (`e47dc7ae`) | This Build (`f1b1eac4`) |
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

- **Exactness:** `64/64` bit-identical `token_sha` on Fibonacci (`fdf5a5a7e5bcb52d`), `p8` (`461ed727d292d70f`), and `--serve` cached-vs-cold (`5c3802fb70b21af4`); `0` compiler warnings, `0` register spills, `0` new TensorFold `lean_check.py` findings (`231 vs 241`).
- **Multimodal validation:** Exact 3-line OCR (`640×640`), exact left-to-right color/shape spatial identification on non-square `800×400` (bilinear pos-emb + 2D RoPE + 2×2 merge), and `0.0 ms` Turn-2 image prefix reuse.

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
- `reasoning_content` + `reasoning_effort` (`low` / `medium` / `high` / numeric)
- OpenAI `tools` + native `<|content_invoke_tool_json|>` → `tool_calls` parsing
- OpenAI `image_url` (`data:image/...;base64,...`, `http(s)://`, `file://`) via native Xe2 XMX `mmproj-F16.gguf`
- Dual GPU prefix KV/conv cache (`ckpt_sys` + `ckpt_last`) for instant Turn-2+ and same-system-prompt TTFT

---

## Repository Layout

- `tensorfold/Containerfile` — Self-contained 3-stage `ubuntu:26.04` build cloning `SergiioB/TensorFold@b70-inkling2-mini-20261121` and applying `tensorfold/patches/*.patch`
- `tensorfold/Makefile` — `build`, `rebuild`, `run`, `stop`, `logs`, `status`, `health`, `smoke`, `bench`, `profile`
- `tensorfold/patches/` — 4 atomic patches (`0001`–`0004`)
- `tensorfold/SERGIO_RECEIPT.md` — Full TensorFold PR-template performance, exactness, and VRAM receipt
- `b70-inkling2-evolution.html` — Standalone interactive 17-slide HTML presentation
- `Containerfile` & `Makefile` — Root Fused Vulkan (`vengeful180-dot/llama.cpp`) comparison setup

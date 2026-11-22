# Project Handoff: Inkling-2-Mini on Intel Arc Pro B70 (32 GB)

## 1. Goal & Context
Serve Thinking Machines' **Inkling-2-Mini** (26B MoE, 128 experts top-8, layers 6–17 looped 3× = 48 layer passes/token, 1M trained context) locally on a single **Intel Arc Pro B70 (32 GB, BMG-G31 / Xe2)** for single-developer agentic use (**Hermes Agent**, which requires `>= 64K` context and sends a ~14.6K-token system prompt on turn 1).

Primary targets:
1. Validate and finalize the working **Fused Vulkan (`vengeful180-dot/llama.cpp`)** setup (and optionally prep a PR for `SergiioB/intel-arc-pro-b70-inference-cookbook`).
2. Chase **70+ tok/s decode** on the B70 (e.g., via SergiioB's **TensorFold XPU** research build `b70-inkling2-mini-20261121`, or further Vulkan/Xe2 kernel tuning).

---

## 2. Host & Environment
- **OS / Runtime:** Ubuntu 26.04 host, rootless **Podman**.
- **GPUs:**
  - `Device 0`: Arrow Lake-S iGPU
  - `Device 1`: **Intel Arc Pro B70 (BMG-G31)** at PCI `0000:04:00.0` → `/dev/dri/renderD129` + `/dev/dri/card0`
- **Active Workspace:** `~/llm-models/Inkling-2-Mini-Unsloth-UD-Q4-K-XL`
  - Model: `Inkling-2-Mini-UD-Q4_K_XL.gguf` (16.41 GiB / 17.63 GB, Unsloth root-folder PR #32695 layout, sha256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`)
  - Container image built: `llama-b70-inkling:vk-fused` (`vengeful180-dot/llama.cpp` @ commit `eb39823`, Ubuntu 26.04, `mesa-vulkan-drivers 26.0.8`, targets: `llama-server`, `llama-bench`, `test-backend-ops`)
- **Separate Archive Dir:** `~/llm-models/Inkling-2-Mini-XQ-Uncensored` (holds `IndrasMirror/Inkling-2-Mini-XQ-Uncensored.gguf` mainline layout + earlier SYCL/mainline Vulkan attempts).

---

## 3. What We Tested & Measured (Intel Arc Pro B70)

| Setup | Model / KV | `pp` (short) | `tg` (short) | `pp` (deep) | `tg` (deep) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `Indras-Mirror/Xolotl` (`sycl/`) | XQ | — | — | — | — | **Dead end:** `xolotl-inkling` is CUDA-only; `sycl/` only builds legacy Strata. |
| Mainline `llama.cpp` SYCL | XQ / `q8_0` | 87→35 t/s | 28.4 t/s | — | 9.8 t/s | 10–30% GPU sawtooth (96 CPU round-trips/tok on strided `SCALE`); crashed >14K (`OP GET_ROWS` int range overflow). |
| Mainline `llama.cpp` Vulkan + PR #32735 | XQ / `q8_0,q4_0` | 509 t/s (2K) | 18.5 t/s | 105 t/s (45K) | 10.3–13.6 t/s | Stable, 99% GPU util, 20.1/31.89 GiB VRAM at 45K, but ~8,100 dispatches/tok + F32 rel-pos bias cliff. |
| SergiioB Patched SYCL (0001–0006 + Graph) | UD-Q4_K_XL | 169–263 t/s (9K) | 29.09 t/s | 150 t/s (39K) | 16.66 t/s (39K) | Reference numbers from `SergiioB/intel-arc-pro-b70-inference-cookbook`. |
| **Fused Vulkan (`vengeful180-dot` `eb39823`)** | UD-Q4_K_XL / `q8_0,q4_0` | 665.41 t/s | 28.89 t/s | 143.94 t/s (32K) | 16.65 t/s (32K) | `llama-bench -p 512 -n 64 -d 0,32768`. |
| **Fused Vulkan (`vengeful180-dot` `eb39823`)** | **UD-Q4_K_XL / `f16,f16`** | **745.35 t/s** | **31.50 t/s** | **319.08 t/s (32K)** | **25.84 t/s (32K)** | **Current best:** beats 6-patch SYCL across every metric on stock Mesa ANV. |

---

## 4. Key Gotchas Already Solved
1. **Ubuntu 26.04 Vulkan ICD path:** Use `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json` (not `intel_icd.x86_64.json`).
2. **Rootless Podman GPU permissions:** Use `--group-add keep-groups` (numeric `--group-add $RENDER_GID` breaks in rootless userns).
3. **Ubuntu 26.04 Vulkan build deps:** `spirv-headers` must be installed alongside `libvulkan-dev` and `glslc`.
4. **GGUF layout split:** `vengeful180-dot/llama.cpp` and SergiioB's patched SYCL use **Unsloth's root-folder** GGUF (PR #32695 layout). Mainline `llama.cpp >= b12623` uses PR #32694/#32704 layout (`Mainline_Support` / `XQ`). They are not interchangeable.
5. **KV cache on Vulkan/B70:** `f16` KV is drastically faster than `q8_0`/`q4_0` at 32K (+55% decode, +122% prefill) because it avoids software dequant in Vulkan attention and feeds `KHR_coopmat`.
6. **Hermes Agent:** Requires `CTX >= 65536`. Inkling-2-Mini defaults to `reasoning_effort: 0.9` which can burn >2,048 thinking tokens before answering; consider tuning `reasoning_effort` or `max_tokens`.

---

## 5. Immediate Next Objectives
1. **Finalize & verify Fused Vulkan (`f16` KV):**
   - Confirm `Makefile` uses `--cache-type-k f16 --cache-type-v f16` (and update `bench` target to match).
   - Run `make test-ops` (`test-backend-ops test -b Vulkan0`) and verify all 7 custom fused Inkling ops pass on Mesa ANV.
   - Run `make run CTX=65536 && make smoke` and confirm Hermes Agent works end-to-end.
2. **Chase 70+ tok/s on B70:**
   - Fetch/inspect `https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/inkling2-mini/README.md` for the **TensorFold XPU** research build (`b70-inkling2-mini-20261121`) and build a clean `Containerfile` + `Makefile` for it to benchmark head-to-head against Fused Vulkan.
   - Optionally investigate whether `vengeful180-dot`'s Vulkan shaders (`fp4: 0`, RDNA3 workgroup tuning) can be tuned for Intel Xe2 / Mesa ANV (`warp size: 32`, `KHR_coopmat`).
3. **Optional Cookbook PR:**
   - If `make test-ops` passes cleanly, run cold-prompt matched benchmarks (`--no-cache-prompt --cache-ram 0` at short / ~9K / ~39K, logging commit SHA `eb39823`, Mesa `26.0.8`, GGUF sha256, and B70 power/temp) to prepare a PR for `SergiioB/intel-arc-pro-b70-inference-cookbook`.

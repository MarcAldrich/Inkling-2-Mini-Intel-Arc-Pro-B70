# TensorFold XPU (`xpu-inkling2`) — Intel Arc Pro B70 Performance & Accuracy Receipt

> **Target upstream:** [`SergiioB/TensorFold`](https://github.com/SergiioB/TensorFold)  
> **Base tag / SHA:** `b70-inkling2-mini-20261121` (`e47dc7ae8e06c9cdefb846e01cf4671bad6cb3db`)  
> **Branch / Head SHA:** `b70-xe2-xmx-dp4a` (`1ff69191`, 5 atomic commits exported in `tensorfold/patches/`)  
> **PR status note:** `zig/src/families/inkling2/` and `zig/kernels/xpu/inkling2/` currently exist only on tag `b70-inkling2-mini-20261121` and are not yet on `origin/main` (`1a564489`); `SergiioB/TensorFold` also has `has_issues: false` and 0 open/closed community PRs at time of measurement. Patches and this complete receipt are formatted directly to `.github/pull_request_template.md` so they can be applied cleanly on top of `b70-inkling2-mini-20261121` or stacked on the `inkling2` PR once opened.

---

## 1. What This Changes (5 Atomic Commits)

1. **`804d71ac` — `xpu-inkling2: Xe2 XMX DPAS prefill/attn, DP4A decode, async pipeline`** (`ink2.cl`, `xpu_main.zig`)
   - **Async pipelined decode:** `ink2_embed_dev` reads the next token ID directly from device memory with a 2-slot host-pinned event ring (`stepV2Async` / `waitSlot`; `INK2_SYNC=1` restores sync).
   - **Vectorized decode control/mix/router + BF16 loop controllers:** vectorized `vload4` `ink2_mix3` / `ink2_mix3n`, split-K `ink2_ctl_down_k`, 0-barrier 1-subgroup top-8 `ink2_router_k`, and BF16 `loop_down4`/`loop_gain4`/`loop_forget4` weights (including `INK2_OLDSHAPE=1` kernels).
   - **Hardware DP4A Q8_0 decode + fused Q8_1 quant:** `dot_acc_sat` kernels (`gemv_bigx4_dp4a`, `gemv_conv2s_dp4a`, `gemv_q8_4_dp4a`, `down_q8_dp4a`) with 0-barrier warp-shuffle `quant4_32` fused into `norm`, `mix3n`, `attn2`, and `attn_merge` (eliminating 101 of 105 `quant_q8` launches/step; `INK2_NODP4A=1` restores FP32).
   - **32-Xe-core exact wave alignment + activation hoisting:** `KERNEL(64)` (`C/8 = 288 WGs = 9.0` exact waves on 32 Xe-cores) on `down_nvfp4d`, `down_q8_dp4a`, and `gemv_conv2s_dp4a`, plus hoisted `vload16` activation loads (`nv_dot32_v`) in `gateup_nvfp4c` and `down_nvfp4d`.
   - **Xe2 XMX DPAS prefill (`TMAX=256`) + FlashAttention + Dual 128/256-GRF:**
     - Exact bitwise NVFP4 E2M1 → FP16 unpack (`nv_pair`, `val * 2^-14`, native subnormal support) mapped 1:1 to `k=16` `intel_sub_group_f16_f16_matrix_mad_k16` (`gateup_nvfp4_xmx`, `down_nvfp4_xmx`) plus Q8_0 XMX (`gemm_q8_xmx`, `gateup_q8_xmx`, `down_q8_xmx`).
     - Dual-module GRF dispatch: 4-accumulator `gateup_*_xmx` load from `kbin.ink2_256` (`0` spills vs `2.1–5.8 KB` spills on 128-GRF); all decode and lighter kernels stay on `kbin.ink2` (128-GRF, 2× thread occupancy).
     - `ink2_attn_b_xmx`: Xe2 XMX DPAS FlashAttention (`15` DPAS columns = 5 tokens × 3 GQA heads/WG, 64-key tiles with zero wrap on `ring=1280`, hardware 2D block reads `16b_8r16x1c` + `transform_16b_16r16x1c`, uniform `has_rel` skip for `dist >= 1024`, online sink softmax `m=0, l=1`).
     - Zero-scratch online-softmax fallback `ink2_attn_b` (eliminates `scratch_b`, saving `1.6 GB` @ 64K and enabling up to `384K` context on a single 32 GiB B70; `INK2_NOXMX=1` restores scalar prefill).
     - 128-thread parallel `ink2_route_lists`, 2D-grid `gemv_f32_b`, and 8-token tiled `ctl_down_b` / `ctl_out_b`.

2. **`f1e74aa9` — `xpu-inkling2: bit-exact dual GPU prefix KV/conv checkpoint cache in --serve`** (`xpu_main.zig`)
   - Adds `PrefixCkpt` (`ckpt_sys` saved before the first `<|message_user|>` token when `>= 256` tokens, and `ckpt_last` saved after full prompt prefill; `INK2_NOPREFIX=1` disables).
   - Each checkpoint snapshots `hist2` (`7.67 MB`) and the 40 sliding-window KV rings (`209.7 MB`, total `~217 MB` each / `~435 MB` for both); the 8 global KV caches remain append-only in-place with strict `p0 < ckpt.len` invalidation.
   - Resumed prompts finish via `prefillRange` so cached and cold executions produce **bit-identical `token_sha`**.

3. **`435143c5` — `inkling2_serve: OpenAI tool_calls, reasoning_effort, and --host support`** (`tools/xpu/inkling2_serve.py`)
   - Passes `tools` and `reasoning_effort` into `chat_template.jinja`, canonicalizes JSON-string assistant `tool_calls` arguments to dicts, parses `<|content_invoke_tool_json|>` into OpenAI `tool_calls` (streaming & non-streaming, `finish_reason="tool_calls"`), adds `--host` (default `127.0.0.1`), and tolerates standard OpenAI `top_p` fields.

4. **`f1b1eac4` — `xpu-inkling2: native Xe2 XMX vision tower (mmproj-F16 inkling2v)`** (`vit.cl`, `xpu_vit.zig`, `xpu_inkling2.zig`, `ink2.cl`, `xpu_main.zig`, `inkling2_serve.py`)
   - Implements the full 23-layer `inkling2v` ViT (`D=1024`, `FF=4096`, 16 heads, `d_head=64`, bilinear 32×32 pos-emb, 2D RoPE GGML normal, LayerScale, `GELU_ERF`, 2×2 spatial merge, 2-layer `4096→4096→2304` projector) natively on Intel Xe2 XMX DPAS (`0` warnings, `0` spills).
   - Injects `[N_img, 2304]` image embeddings directly into `S[0]` without `embed_norm` (`ink2_embed_mix_b`) using negative `0x80000000 + offset` token IDs, preserving multi-turn `PrefixCkpt` caching across image prompts (`+0.79 GB` VRAM including a 16,384-token image pool → `27.40 GB` total at `CTX=262144`).
   - Verified end-to-end over OpenAI `/v1/chat/completions`:
     - **OCR (`640×640`, 256 img tok):** `"ARC PRO B70 / 32GB GDDR6 / 84.9 TOK/S"` → **100% exact**, `52.2 ms` warm ViT (`72.7 ms` cold), `82.6 tok/s` decode.
     - **Non-square geometry (`800×400`, 200 img tok):** Red circle left, green triangle center, blue square right → **`"Red circle, green triangle, blue square."`** (`36.4 ms` ViT).
     - **Multi-turn image cache (Turn 2 on same `800×400`):** `"Green"` with **`vision_ms = 0.0 ms`** and **`prompt_ms = 71.2 ms`** (`82.2 tok/s` decode).
     - **Resolution latency sweep:** `320×320` (`64` tok) = `12.7 ms`, `640×640` (`256` tok) = `52.2 ms`, `1280×720` (`576` tok) = `126.0 ms`, `1920×1080` (`1,296` tok) = `463.7 ms`.

5. **`1ff69191` — `xpu-inkling2: fix Gumbel float32 +INF bug, add top_p/top_k/min_p, zero XMX KV tail`** (`ink2.cl`, `xpu_main.zig`, `inkling2_serve.py`)
   - **Critical upstream sampling fix (`ink2_sample_part`):** In base `e47dc7ae`, `((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f)` rounds `(float)16777215 + 0.5f` up to `16777216.0f` (`u = 1.0f`) under IEEE-754 round-to-nearest-even, producing `-log(-log(1.0f)) = +INFINITY` and overriding all logits with a uniformly random token across the 200,058-token vocabulary once every ~84 tokens (`1.192%/tok`). Replaced with strictly finite 32-bit `V = 1 - U` Gumbel sampling (`u1 in [2^-33, 1 - 2^-24]`, Taylor `-log1p(-u1)` for `u1 < 1e-4`, `KS = 0.00064` over 2M draws).
   - **Exact `top_p` / `top_k` / `min_p` sampling:** Added `s <max_new> <temp> <top_p> <top_k> <min_p> <seed> <ids...>` protocol (backwards-compatible with legacy format), keeping the 0-overhead 2-slot async GPU Gumbel path when `top_p=1, top_k=0, min_p=0` and using exact `f64` host filtering (`~77.6 tok/s`) when truncation is requested; `inkling2_serve.py` validates and forwards `top_p`, `top_k`, `min_p`.
   - **Upstream `ink2_mix` OOB fix + `INK2_V1=1` BF16 controller fix:** Guarded `d2`/`out[2*C+c]` with `if (nq > 2)` (previously `if (nq > 1)` wrote past `2*C` `a.msg_a`/`a.msg_k` and read past `m.loop_q`), and updated `stepV1` `control()` to BF16 kernels (`INK2_V1=1` now matches pure FP32 `8/8`).
   - **XMX KV padding hygiene:** `ink2_prep_b` zeroes `<16`-row tail padding of `Kc`/`Vc` (with `RING = 1344 = 21*64`) so 16-row Xe2 2D block reads in `ink2_attn_b_xmx` never touch uninitialized FP16 padding.
   - **Regression suite (`tensorfold/test_quality.py`):** Verified `16/16` short decode parity (`mean_KL=0.00040`), bit-exact `PrefixCkpt` at `2,816 > RING` (`max_diff=0.00`), `0` stray foreign-script chars across `4,096` sampled tokens (`T=1.0`, `top_p=1.0 & 0.95`, including former `+INF` trigger seeds `1001`/`1002`), and live HTTP Hermes multi-turn `tool_calls` + SSE streaming + vision.

---

## 2. Environment & Checks on Head `1ff69191`

- **Base SHA:** `e47dc7ae8e06c9cdefb846e01cf4671bad6cb3db` (`b70-inkling2-mini-20261121`)
- **Head SHA:** `1ff69191` (stack: `804d71ac` → `f1e74aa9` → `435143c5` → `f1b1eac4` → `1ff69191`)
- **Machine / GPU / OS:** Linux `7.0.0-44-generic` `x86_64`, `30 GiB` host RAM, **Intel Arc Pro B70 (BMG-G31 / Xe2, PCI `8086:e223`, 32 GiB GDDR6, 256-bit ~608 GB/s, 32 Xe-cores)**
- **Containerized Toolchain (`ubuntu:26.04`, zero host build deps):**
  - Zig: `0.17.0`
  - Compiler: `intel-ocloc 26.05.37020.3-1`, `libigc2 2.28.4-4`, `libigdgmm12 22.9.0+ds1-1`
  - Runtime: `libze1 1.28.2-2`, `libze-intel-gpu1 26.05.37020.3-1`, `libigc2 2.28.4-4`, `libigdfcl2 2.28.4-4`
- **Checkpoint:** `Inkling-2-Mini-UD-Q4_K_XL.gguf` (`17,629,759,456` bytes, sha256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`), tokenizer/template `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`
- **Checks:**
  - `zig fmt --check build.zig zig`: **PASS**
  - `git diff --check e47dc7ae..HEAD`: **PASS**
  - `python3 tools/zig/lean_check.py` / `tools/pr_check.py --base e47dc7ae`: **base `241` → head `231` (`0` new findings, `-10` net)** *(2 inherited >600-line file errors on `ink2.cl` and `xpu_main.zig`, both already >1500 lines on `e47dc7ae`)*
  - `zig build test-inkling2-host`: **`24/28` passed, `4` skipped, `0` failed** (`80 ms`)
  - `ocloc` (`bmg`, 128-GRF and 256-GRF): **`0` register spills, `0` warnings**
  - Author/committer check (`git log --format='%an <%ae>'`): `Marc Aldrich <7917445+MarcAldrich@users.noreply.github.com>` across all 3 commits; no local paths or AI trailers.
- **Platforms not run:** Metal, CUDA, HIP, Windows (all changes are scoped exclusively to `zig/kernels/xpu/inkling2/`, `zig/src/families/inkling2/xpu_main.zig`, and `tools/xpu/inkling2_serve.py`).

---

## 3. Exactness & Parity Verification

| Test | Comparison | Result | Hash / Error |
|---|---|---|---|
| Fibonacci prompt (`29` tok, 64 greedy tokens) | `HEAD` vs `INK2_NOXMX=1 INK2_NODP4A=1 INK2_SYNC=1` | **64/64 exact match** | `token_sha = fdf5a5a7e5bcb52d` (both) |
| Synthetic `[1, 200001, 1000..6000]` (64 greedy tokens) | `HEAD` vs `INK2_NOXMX=1 INK2_NODP4A=1 INK2_SYNC=1` | **64/64 exact match** | `token_sha = 461ed727d292d70f` (both) |
| `--serve` Turn-2 Prefix Cache (`542` tok, XMX + DP4A enabled) | Cached Turn-2 (`74.6 ms`) vs Cold `INK2_NOPREFIX=1` (`613.5 ms`) | **64/64 exact match** | `token_sha = 5c3802fb70b21af4` (both) |
| Sliding-window ring wrap (`1,536` tok `> RING=1280`) | `ink2_attn_b_xmx` vs `INK2_NOXMX=1` at `pos=255 / 1280 / 1535` | **Top-4 logits identical** | Visits `v0–v6` rel L2 = `1.59e-4 – 2.98e-4` |

---

## 4. Performance Comparison (Base `e47dc7ae` vs Head `435143c5`)

### A. Decode & Prefill Headline Comparison (Same B70 & GGUF)

| Workload | Fused Vulkan (`llama.cpp` `eb39823`) | Base TensorFold XPU (`e47dc7ae`) | Head (`435143c5`) | Speedup vs Base XPU |
|---|---:|---:|---:|---:|
| Short decode (`p8`, 63 steps) | `31.74` (`34.61` no-async) | `71.00` (`73.40` warm) | **`84.94 tok/s`** (`87.4` 15-step) | **`+19.6%`** |
| Decode GPU busy / launches (`INK2_PROFILE=1`) | `30.55 ms` | `13.174 ms` (`533` launches) | **`11.692 ms`** (`537` launches) | **`-11.3%` GPU time** |
| Real prompt decode (`fib`, 63 steps) | — | — | **`82.84 tok/s`** | — |
| 512-tok cold prefill | — | `116.2 tok/s` (`4.41 s`) | **`916.8 tok/s`** (`0.558 s`) | **`7.89×`** |
| 14.6K Hermes Turn-1 cold prefill | — | `~85 tok/s` (`~170 s`) | **`873.5 tok/s`** (`16.71 s`) | **`~10.2×`** |
| 64K (`65,536` tok) cold prefill | `~10 min` (`~600 s`) | OOM / `>10 min` | **`782.0 tok/s`** (**`83.81 s` / `1.40 min`**) | **`~7.2×` vs Vulkan** |
| 128K (`131,072` tok) cold prefill | OOM on 32 GiB | OOM | **`689.6 tok/s`** (**`190.07 s` / `3.17 min`**) | **2× Vulkan max ctx in <1/3 time** |
| 20.3K Hermes Turn-2 / New-Chat TTFT (`--serve`) | cold | cold (`~240 s`) | **`94.8 ms` / `89.7 ms`** (vs `24.15 s` cold) | **`255× – 269×`** |

### B. Full Measured Context Scaling Sweep (`HEAD 435143c5`, `CTX=65536` & `CTX=262144`)

| Prompt Length | Cold Prefill Time | Cold Prefill Speed | Decode Speed (63 steps) |
|---:|---:|---:|---:|
| `256` | `0.280 s` | `914.2 tok/s` | `83.84 tok/s` |
| `512` | `0.558 s` | `916.8 tok/s` | `82.94 tok/s` |
| `2,048` | `2.259 s` | `906.5 tok/s` | `79.60 tok/s` |
| `4,096` | `4.554 s` | `899.5 tok/s` | `77.23 tok/s` |
| `8,192` | `9.204 s` | `890.0 tok/s` | `73.02 tok/s` |
| `14,600` | `16.714 s` | `873.5 tok/s` | `67.06 tok/s` |
| `32,768` (32K) | `39.082 s` | `838.4 tok/s` | `54.82 tok/s` |
| `65,536` (64K) | **`83.808 s` (`1.40 min`)** | **`782.0 tok/s`** | **`41.23 tok/s`** |
| `131,072` (128K) | **`190.069 s` (`3.17 min`)** | **`689.6 tok/s`** | **`27.56 tok/s`** |

*(Quadratic fit across 32K/64K/128K: `t(N) = 1.11e-3·N + 2.60e-9·N²`, `R² = 0.99999` → projects ~`7.7 min` cold prefill at 256K and ~`10.7 min` cold prefill at 320K — **5× Vulkan's 64K context in the same ~10 min window**, with Turn-2+ cached TTFT `< 100 ms`.)*

### C. High-Context VRAM Fit on 32 GiB Intel Arc Pro B70

Fixed weights = `17.56 GB`, fixed activations + 40 sliding rings (`RING=1280`) = `~0.52 GB`, marginal global KV + split-attn = `32.28 KB/tok` (`2.17 GB` per 64K); `--serve` dual prefix cache adds `+0.43 GB` fixed:

| `CTX` | Runner Card VRAM (`d.bytes`) | `--serve` Card VRAM (+dual cache) | Short Decode (15 steps) |
|---:|---:|---:|---:|
| `65,536` (64K) | `20.11 GB` (`18.73 GiB`) | `20.54 GB` (`19.13 GiB`) | `87.34 tok/s` |
| `131,072` (128K) | `22.28 GB` (`20.75 GiB`) | `22.71 GB` (`21.15 GiB`) | `87.43 tok/s` |
| `262,144` (256K) | `26.61 GB` (`24.78 GiB`) | `27.04 GB` (`25.18 GiB`) | `87.38 tok/s` |
| **`327,680` (320K, live default)** | **`28.78 GB` (`26.80 GiB`)** | **`29.21 GB` (`27.20 GiB`)** | **`87.48 tok/s`** (`80.90` HTTP smoke) |
| `360,448` (352K) | `29.86 GB` (`27.81 GiB`) | `30.29 GB` (`28.21 GiB`) | `87.32 tok/s` |
| **`393,216` (384K max verified)** | **`30.94 GB` (`28.82 GiB`)** | `31.37 GB` (`29.22 GiB`) | **`87.25 tok/s`** |

### D. Per-Kernel GPU Profile (`INK2_PROFILE=1`, `HEAD 435143c5`, 64 Steps, `CTX=4096`)

```
GPU busy 11.692 ms a step, 537 launches a step (85.5 tok/s GPU-only ceiling)
  gemv_bigx4_dp4a   2.712 ms/step  23.2 %   48.0 launches/step   56.49 us each
  gateup_nvfp4c     2.372 ms/step  20.3 %   44.0 launches/step   53.91 us each
  down_nvfp4d       1.240 ms/step  10.6 %   44.0 launches/step   28.18 us each
  attn2             0.860 ms/step   7.4 %   48.0 launches/step   17.92 us each
  gemv_conv2s_dp4a  0.855 ms/step   7.3 %   48.0 launches/step   17.80 us each
  gemv_q8_4_dp4a    0.851 ms/step   7.3 %    1.0 launches/step  850.66 us each  (578.3 GB/s = 95.1% B70 peak)
  mix3              0.815 ms/step   7.0 %   96.0 launches/step    8.49 us each
  gateup_q8         0.451 ms/step   3.9 %    4.0 launches/step  112.73 us each
  router_k          0.419 ms/step   3.6 %   48.0 launches/step    8.72 us each
  mix3n             0.288 ms/step   2.5 %   24.0 launches/step   12.01 us each
  ctl_down_k        0.269 ms/step   2.3 %   48.0 launches/step    5.61 us each
  ctl_out2          0.227 ms/step   1.9 %   48.0 launches/step    4.74 us each
  down_q8_dp4a      0.199 ms/step   1.7 %    4.0 launches/step   49.79 us each
  norm              0.092 ms/step   0.8 %   25.0 launches/step    3.70 us each
  argmax_final      0.018 ms/step   0.2 %    1.0 launches/step   17.95 us each
  sample_part       0.011 ms/step   0.1 %    1.0 launches/step   10.59 us each
  quant_q8          0.007 ms/step   0.1 %    4.0 launches/step    1.83 us each
  embed             0.007 ms/step   0.1 %    1.0 launches/step    6.66 us each
```

---

## 5. Additional Patches Needed to Land `inkling2` Upstream on `TensorFold/main`

Because `b70-inkling2-mini-20261121` is currently a standalone research tag not merged into `main`, landing these patches into `TensorFold/main` would additionally need:
1. **Base `inkling2` family merge:** Cherry-pick/merge the 24 `inkling2` files from tag `b70-inkling2-mini-20261121` (`e47dc7ae`) onto `main` first (or apply our 3 patches directly onto Sergio's working `inkling2` branch).
2. **600-line lean file split (for `pr_check.py` on `main`):** Split `ink2.cl` (currently 3,230 lines; 2,247 on base) and `xpu_main.zig` (currently 1,808 lines; 1,527 on base) into `<= 600`-line modules (e.g., `ink2_decode.cl`, `ink2_dp4a.cl`, `ink2_xmx.cl`, `ink2_attn_xmx.cl`, plus `xpu_loader.zig`, `xpu_prefix.zig`, `xpu_serve.zig`).
3. **Environment registry & kernel contract:** Register any retained `INK2_*` runtime switches in `zig/src/platform/environment.zig` (if promoted to `TF_*` server flags) and register the new Xe2 XMX/DP4A kernels in `docs/kernel-contract.md` / XPU contract tests once `xpu-inkling2` is integrated into `tensorfold-native`.
4. **Native Zig server integration (optional):** Wire `PrefixCkpt` into TensorFold's shared core prompt-cache store (`TENSORFOLD_PROMPT_CACHE_*`) when `inkling2` graduates from the standalone `xpu-inkling2` runner into `tensorfold-native`.
5. **CHANGELOG.md:** Add an `## Unreleased` bullet naming the PR number once opened.

---

## 6. Applying the Exported Patches

```sh
git clone --branch b70-inkling2-mini-20261121 https://github.com/SergiioB/TensorFold.git
cd TensorFold
git am /path/to/tensorfold/patches/0001-*.patch \
       /path/to/tensorfold/patches/0002-*.patch \
       /path/to/tensorfold/patches/0003-*.patch \
       /path/to/tensorfold/patches/0004-*.patch \
       /path/to/tensorfold/patches/0005-*.patch
```

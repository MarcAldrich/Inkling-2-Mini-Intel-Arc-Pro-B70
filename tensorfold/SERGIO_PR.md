# TensorFold PR Package — `xpu-inkling2` Intel Arc Pro B70 (Xe2 XMX, DP4A, Vision, Prefix Cache & Sampling Fixes)

> **How to submit this PR cleanly to `SergiioB/TensorFold`:**
> 1. Fork [`SergiioB/TensorFold`](https://github.com/SergiioB/TensorFold) on GitHub to `MarcAldrich/TensorFold`.
> 2. Push the existing clean local 6-commit branch `b70-xe2-xmx-dp4a` (`e47dc7ae` → `b343e77e`, verified `git am` + `pr_check.py`) to your fork:
>    ```sh
>    cd tensorfold/src
>    git remote add fork git@github.com:MarcAldrich/TensorFold.git
>    git push -u fork b70-xe2-xmx-dp4a
>    ```
> 3. **Important branch-target note:** `b70-inkling2-mini-20261121` (`e47dc7ae`) is currently a git **tag** not yet merged onto `SergiioB/TensorFold:main` (`1a564489`). Until Sergio creates an `inkling2` branch or merges the tag onto `main`, opening a GitHub PR targeting `main` will include the base tag's `inkling2` files too. You can either:
>    - Push both a base ref (`git push fork e47dc7ae:refs/heads/b70-inkling2-mini-20261121`) and `b70-xe2-xmx-dp4a` to `MarcAldrich/TensorFold` (gives a clean 6-commit compare URL `https://github.com/MarcAldrich/TensorFold/compare/b70-inkling2-mini-20261121...b70-xe2-xmx-dp4a`), or
>    - Open the PR once an `inkling2` target branch exists on upstream, pasting the PR body below verbatim!

---

## Copy-Paste PR Title

```text
xpu-inkling2: Xe2 XMX prefill/attn/vision, DP4A decode, dual prefix cache, and Gumbel +INF sampling fix
```

---

## Copy-Paste PR Description (matches `.github/pull_request_template.md`)

```markdown
## What this changes

Optimizes `xpu-inkling2` for Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GiB) and fixes critical sampling, OOB, and streaming disconnect bugs across 6 atomic commits stacked cleanly on tag `b70-inkling2-mini-20261121` (`e47dc7ae`):

1. **`804d71ac` — Xe2 XMX DPAS prefill/attn, DP4A decode, async pipeline** (`ink2.cl`, `xpu_main.zig`):
   - Async 2-slot event-ring decode (`ink2_embed_dev`, `INK2_SYNC=1` fallback); vectorized `vload4` `mix3`/`mix3n`, split-K BF16 loop controllers, 0-barrier 1-subgroup top-8 `router_k`; hardware DP4A Q8_0 decode with fused warp-shuffle Q8_1 quant (`INK2_NODP4A=1` fallback); 32-Xe-core wave alignment (`KERNEL(64)`).
   - Xe2 XMX DPAS prefill (`TMAX=256`) for NVFP4 (exact E2M1→FP16 unpack) and Q8_0, dual 128/256-GRF module dispatch (`0` spills), `ink2_attn_b_xmx` FlashAttention with Xe2 2D block reads (`INK2_NOXMX=1` fallback), and zero-scratch online-softmax fallback enabling up to 384K context on 32 GiB.
2. **`f1e74aa9` — Bit-exact dual GPU prefix KV/conv cache in `--serve`** (`xpu_main.zig`):
   - `PrefixCkpt` (`ckpt_sys` + `ckpt_last`, `INK2_NOPREFIX=1` disables) snapshotting `hist2` + 40 sliding KV rings (`~217 MB` each) with append-only global KV and strict `p0 < ckpt.len` invalidation; resumed prompts finish via `prefillRange` for bit-identical `token_sha`.
3. **`435143c5` — OpenAI `tool_calls`, `reasoning_effort`, and `--host` support** (`tools/xpu/inkling2_serve.py`):
   - Wires `tools` and `reasoning_effort` into `chat_template.jinja`, parses `<|content_invoke_tool_json|>` into OpenAI `tool_calls` (streaming & non-streaming), and adds `--host`.
4. **`f1b1eac4` — Native Xe2 XMX vision tower (`mmproj-F16` / `inkling2v`)** (`vit.cl`, `xpu_vit.zig`, `xpu_inkling2.zig`, `ink2.cl`, `xpu_main.zig`, `inkling2_serve.py`):
   - Full 23-layer `inkling2v` ViT + 2×2 spatial merge + 2-layer projector on Xe2 XMX DPAS (`0` spills, `+0.79 GB` VRAM incl. 16K-token image pool), `ink2_embed_mix_b` injection preserving `PrefixCkpt` multi-turn image caching, and OpenAI `image_url` handling.
5. **`1ff69191` — Fix Gumbel float32 `+INF` bug, add `top_p`/`top_k`/`min_p`, fix `ink2_mix` OOB, zero XMX KV tail** (`ink2.cl`, `xpu_main.zig`, `inkling2_serve.py`):
   - Fixes upstream `ink2_sample_part` float32 rounding bug (`(float)0xFFFFFF + 0.5f → 16777216.0f → u=1.0 → -log(-log(1.0)) = +INF`, injecting a random vocab token ~1/84 tokens at `T > 0`) with strictly finite 32-bit `V = 1 - U` Gumbel sampling (`KS = 0.00064` over 2M draws) plus exact `f64` `top_p`/`top_k`/`min_p` support.
   - Fixes upstream `ink2_mix` OOB read/write when `nq == 2` (`if (nq > 2)` instead of `if (nq > 1)`) and zeroes `<16`-row KV tail padding (`RING = 1344`) for Xe2 2D block reads.
6. **`b343e77e` — Drain runner on mid-stream client disconnect, bump default `max_tokens` to `16384`** (`tools/xpu/inkling2_serve.py`):
   - Keeps reading runner stdout until `DONE` if an SSE client drops mid-stream so the next request never hits `Protocol desync`.

- Issue: n/a (`SergiioB/TensorFold` has issues disabled; `inkling2` currently lives on tag `b70-inkling2-mini-20261121` rather than `main`).
- Stacked on: tag `b70-inkling2-mini-20261121` (`e47dc7ae8e06c9cdefb846e01cf4671bad6cb3db`) — 6 atomic commits (`804d71ac` → `f1e74aa9` → `435143c5` → `f1b1eac4` → `1ff69191` → `b343e77e`). Happy to split or rebase onto an `inkling2` branch / `main` once the base family lands.
- `atc overlap --json`: n/a (target `inkling2` files exist only on tag `b70-inkling2-mini-20261121`).

## Checks on this head

- Head commit: `b343e77e41761a2b72a47e614d055ffb37455cf8`
- Lean checker, `python3 tools/zig/lean_check.py`, on the base and on this head: base `241` → head `231` (**`0` new findings**, `-10` net). `python3 tools/pr_check.py --base e47dc7ae` reports `6 changed files; lean 231 vs 241, 0 new` (2 inherited `>600`-line file errors on `ink2.cl` and `xpu_main.zig`, both already `>1500` lines on `e47dc7ae`).
- Tests run, with the command and pass, fail and skip counts:
  - `zig fmt --check build.zig zig`: PASS
  - `git diff --check e47dc7ae..HEAD`: PASS
  - `zig build test-inkling2-host`: `24` passed, `0` failed, `4` skipped (`80 ms`)
  - `ocloc` (`bmg`, 128-GRF and 256-GRF, `ink2.cl` + `vit.cl`): `0` register spills, `0` warnings
  - `test_quality.py --mode runner` & `--mode http`: ALL PASS (16/16 parity `mean_KL=0.00040`, bit-exact `PrefixCkpt` at 2,816 tok `max_diff=0.00`, `0` stray foreign-script tokens across `4,096` sampled tokens at `T=1.0`, clean multi-turn `tool_calls` + SSE + vision OCR/geometry + `0.0 ms` Turn-2 image cache).
  - 8+ hour live `--serve` soak test (`CTX=262144` + `mmproj-F16.gguf`): `140+` multi-turn coding/tool-calling/vision calls, `19.9M+` characters scanned with `0` stray/garbled characters and `0` disconnect desyncs.
- CI: Not yet run on GitHub Actions (branch not yet targeting `main`).
- Not run, and why: Metal, CUDA, HIP, Windows — all changes are scoped strictly to `zig/kernels/xpu/inkling2/`, `zig/src/families/inkling2/`, `zig/build/xpu_inkling2.zig`, and `tools/xpu/inkling2_serve.py`.

## Receipt

- Main against branch: Measured against base tag `b70-inkling2-mini-20261121` (`e47dc7ae`) on the same Intel Arc Pro B70 and GGUF (cannot run against `main` `1a564489` because `inkling2` is not yet on `main`).
- Boxes it reaches: Intel XPU Level Zero (`xpu-inkling2` standalone runner + `tools/xpu/inkling2_serve.py`).
- New switch? Default, accepted values, profile row:
  - Diagnostic fallback switches on `xpu-inkling2`: `INK2_SYNC=1` (disable async decode ring), `INK2_NODP4A=1` (disable DP4A Q8_0 decode), `INK2_NOXMX=1` (disable Xe2 XMX prefill/attn), `INK2_NOPREFIX=1` (disable dual GPU prefix cache). All default off (fast path enabled).
  - Server flags on `inkling2_serve.py`: `--host` (default `127.0.0.1`), `--mmproj <path>` (optional, enables vision).
- Environment:
  - TensorFold base: `b70-inkling2-mini-20261121` (`e47dc7ae`), head `b343e77e`
  - Toolchain (containerized `ubuntu:26.04`): Zig `0.17.0`, `intel-ocloc 26.05.37020.3-1`, `libigc2 2.28.4-4`, `libze1 1.28.2-2`, `libze-intel-gpu1 26.05.37020.3-1`
  - Machine & GPU: Linux `7.0.0-44-generic` `x86_64`, `30 GiB` host RAM, Intel Arc Pro B70 (BMG-G31 / Xe2, PCI `8086:e223`, 32 GiB GDDR6, 32 Xe-cores)
  - Checkpoint: `unsloth/Inkling-2-Mini-GGUF` `Inkling-2-Mini-UD-Q4_K_XL.gguf` (`17,629,759,456` bytes, sha256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`) + `mmproj-F16.gguf`, tokenizer/template `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`
- Paths and platforms the change reaches: Intel Xe2 Level Zero XPU (`xpu-inkling2` + `inkling2_serve.py`, all run on Arc Pro B70). Not run on non-Xe2 Intel GPUs or non-XPU backends.
- Exactness:
  - Fibonacci (29 tok, 64 greedy): `HEAD` vs `INK2_NOXMX=1 INK2_NODP4A=1 INK2_SYNC=1` → **64/64 exact match**, `token_sha = fdf5a5a7e5bcb52d`
  - Synthetic `[1, 200001, 1000..6000]` (64 greedy): `HEAD` vs scalar fallback → **64/64 exact match**, `token_sha = 461ed727d292d70f`
  - `--serve` Turn-2 Prefix Cache (`542` tok): cached (`74.6 ms`) vs cold `INK2_NOPREFIX=1` (`613.5 ms`) → **64/64 exact match**, `token_sha = 5c3802fb70b21af4`
  - Sliding-window wrap (`1,536` tok `> RING=1280` & `2,816` tok `> RING=1344`): top-4 logits identical (`rel L2 = 1.59e-4–2.98e-4`), `PrefixCkpt` bit-exact (`max_diff = 0.00`).
- Main against this branch (base `e47dc7ae` vs `b343e77e`, same B70 & GGUF):
  - Decode (`p8`, 63 steps): `71.00` (`73.40` warm) → **`84.94 tok/s`** (`+19.6%`; `11.69 ms` GPU busy; LM-head `578.3 GB/s` = `95.1%` B70 peak)
  - Cold prefill (`512` tok): `116.2 tok/s` (`4.41 s`) → **`916.8 tok/s` (`0.558 s`, `7.89×`)**
  - Cold prefill (`14.6K` tok): `~85 tok/s` (`~170 s`) → **`873.5 tok/s` (`16.71 s`, `10.2×`)**
  - Cold prefill (`64K` / `128K` tok): OOM / `>10 min` → **`782.0 tok/s` (`83.81 s`)** / **`689.6 tok/s` (`190.07 s`)**
  - Cached 20.3K Turn-2 / New-Chat TTFT (`--serve`): `24.15 s` cold → **`94.8 ms` (`255×`) / `89.7 ms` (`269×`)**
  - Max verified context on 32 GiB: **`384K` (`30.94 GB`)** text-only / **`256K` (`27.40 GB`)** with native Xe2 XMX Vision (`52.2 ms` @ 640px, `126 ms` @ 720p, `0.0 ms` Turn-2 image cache).
- Licence of any new weights or drafter: n/a (uses upstream `thinkingmachines/Inkling-2-Mini` / `unsloth/Inkling-2-Mini-GGUF`).

Full profiling table, VRAM fit sweep, and reproduction stack: [`tensorfold/SERGIO_RECEIPT.md`](https://github.com/MarcAldrich/Inkling-2-Mini-Intel-Arc-Pro-B70/blob/main/tensorfold/SERGIO_RECEIPT.md).

<details><summary>Kernel contract (standalone xpu-inkling2 kernels — not yet wired into tensorfold-native)</summary>

- New/updated OpenCL C kernels in `zig/kernels/xpu/inkling2/ink2.cl` and `zig/kernels/xpu/inkling2/vit.cl`:
  - Decode: `ink2_embed_dev`, `ink2_mix3`/`ink2_mix3n`, `ink2_ctl_down_k`, `ink2_router_k`, `gemv_bigx4_dp4a`, `gemv_conv2s_dp4a`, `gemv_q8_4_dp4a`, `down_q8_dp4a`, `ink2_sample_part` (fixed 32-bit `V=1-U` Gumbel).
  - Prefill / Attention: `gemm_q8_xmx`, `gateup_q8_xmx`, `down_q8_xmx`, `gateup_nvfp4_xmx`, `down_nvfp4_xmx`, `ink2_attn_b_xmx`, `ink2_attn_b` (zero-scratch), `ink2_embed_mix_b`, `ink2_prep_b` (KV tail zeroing).
  - Vision (`vit.cl`): Xe2 XMX DPAS F16 GEMM, bidirectional XMX FlashAttention, 2D RoPE, LayerNorm/LayerScale, 2×2 spatial merge.
- Standalone `xpu-inkling2` is not yet registered in `docs/kernel-contract.md` on `b70-inkling2-mini-20261121`; all fast paths have runtime fallback toggles (`INK2_NOXMX=1`, `INK2_NODP4A=1`, `INK2_SYNC=1`, `INK2_NOPREFIX=1`) verified bit-exact / `mean_KL=0.00040` against scalar FP32 reference.
- Chips not run: Intel Alchemist / Xe1, integrated Xe-LPG, NVIDIA, AMD, Apple Silicon.

</details>

## Checklist

- [x] The title and description match the head commit (`b343e77e`), including the receipt and the 6-commit stack.
- [x] New tests (`test_quality.py` Gumbel `+INF` boundary & stray-script coherence, `PrefixCkpt` > RING parity, vision OCR/geometry/cache) fail before the fix and pass after it.
- [x] No personal data, machine names, internal hosts or local paths in the diff or its commits. No AI attribution lines.
- [x] A resumed prompt equals a fresh one (`token_sha` bit-identical), and decode/prefill are strictly faster across every measured context length.
- [x] `lean_check.py` adds `0` new findings (`231` vs `241` base); happy to split `ink2.cl`/`xpu_main.zig` under 600 lines and integrate into `tensorfold-native` once `inkling2` lands on `main`.
```

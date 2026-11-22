#!/usr/bin/env python3
"""Comprehensive quality, numerical parity, sampling, tool-calling, and vision regression suite for Inkling-2-Mini XPU.

Usage:
  # Direct runner gates (inside container with GPU):
  python3 test_quality.py --mode runner --runner /app/bin/xpu-inkling2 --model /models/Inkling-2-Mini-UD-Q4_K_XL.gguf --meta /app/meta

  # Live HTTP server gates (against running inkling2_serve.py on port 8080):
  python3 test_quality.py --mode http --url http://127.0.0.1:8080
"""
import argparse
import base64
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import http.client

import numpy as np


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def ok(msg):
    print(f"PASS: {msg}", flush=True)


def kl_div(p_log, q_log):
    p = np.exp(p_log - p_log.max())
    p /= p.sum()
    q = np.exp(q_log - q_log.max())
    q /= q.sum()
    return float(np.sum(p * (np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(q, 1e-30)))))


def find_stray_scripts(text):
    """Detect stray non-Latin scripts (CJK, Cyrillic, Arabic, Devanagari, Tamil, Hangul, Armenian, etc.) in English output."""
    bad = []
    allowed_punct = set("—–‘’“”…•·×÷±≤≥≠≈∞→←↑↓✓✗©®™°µ§¶")
    for idx, ch in enumerate(text):
        o = ord(ch)
        if o < 128 or ch in allowed_punct:
            continue
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, "UNKNOWN")
        # Allow basic Latin-1 punctuation / accents / emoji symbols, flag foreign letters & weird symbols
        if cat.startswith("L") and not name.startswith("LATIN"):
            bad.append((idx, ch, name))
        elif o > 0x2BFF and not (0x1F300 <= o <= 0x1FAFF):
            bad.append((idx, ch, name))
    return bad


# ---------------------------------------------------------------------------- Gate 1: Pure math check of fixed Gumbel formula
def test_gumbel_math():
    """Verify the 32-bit V=1-U Gumbel formula is strictly finite for all boundary hashes and matches Gumbel(0,1)."""
    def gumbel_f32(h_arr):
        h_f = h_arr.astype(np.float32)
        u1 = np.minimum((h_f + np.float32(0.5)) * np.float32(2.0 ** -32), np.float32(1.0 - 2.0 ** -24))
        e = np.where(u1 < np.float32(1e-4), u1 + np.float32(0.5) * u1 * u1, -np.log1p(-u1))
        return -np.log(e)

    # Critical boundary hashes including old 0xFFFFFF<<8 bug trigger
    edge = np.array(
        [0, 1, 2, 255, 0x7FFFFF, 0x800000, 0xFFFFFF, 0xFFFFFF00, 0xFFFFFF7F, 0xFFFFFF80, 0xFFFFFFFE, 0xFFFFFFFF],
        dtype=np.uint32,
    )
    g_edge = gumbel_f32(edge)
    if not np.all(np.isfinite(g_edge)):
        fail(f"non-finite Gumbel on boundary hashes: {list(zip(edge.tolist(), g_edge.tolist()))}")

    # Monte Carlo KS test against exact Gumbel(0, 1) CDF = exp(-exp(-x)) over 2M hashes
    rng = np.random.default_rng(2026)
    sample = rng.integers(0, 1 << 32, size=2_000_000, dtype=np.uint32)
    g = gumbel_f32(sample)
    if not np.all(np.isfinite(g)):
        fail("non-finite Gumbel in 2M random sample")
    xs = np.sort(g.astype(np.float64))
    cdf_emp = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
    cdf_ref = np.exp(-np.exp(-xs))
    ks = float(np.max(np.abs(cdf_emp - cdf_ref)))
    if ks > 0.0025:
        fail(f"Gumbel KS statistic too large: {ks:.5f}")
    ok(f"Gumbel 32-bit formula strictly finite on all boundaries (min={g.min():.3f}, max={g.max():.3f}), KS={ks:.5f}")


# ---------------------------------------------------------------------------- Gate 2: Direct runner parity & coherence
def run_serve_batch(runner, model, ctx, env_extra, requests, dump=False):
    env = dict(os.environ)
    env.update(env_extra)
    ddir = tempfile.mkdtemp(prefix="qtest_") if dump else None
    if dump:
        env["INK2_DUMP"] = ddir
    p = subprocess.Popen(
        [runner, model, "--serve", str(ctx)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    line = p.stdout.readline().strip()
    if line != "ready":
        err = p.stderr.read()
        p.kill()
        fail(f"runner failed to start: {line!r} stderr={err}")
    outputs = []
    timings = []
    for req in requests:
        if len(req) == 4:
            ids, nnew, temp, seed = req
            cmd = f"{nnew} {temp} {seed} " + " ".join(map(str, ids)) + "\n"
        else:
            ids, nnew, temp, top_p, top_k, min_p, seed = req
            cmd = f"s {nnew} {temp} {top_p} {top_k} {min_p} {seed} " + " ".join(map(str, ids)) + "\n"
        p.stdin.write(cmd)
        p.stdin.flush()
        toks = []
        while True:
            raw = p.stdout.readline()
            if not raw:
                err = p.stderr.read()
                fail(f"runner exited unexpectedly: stderr={err}")
            w = raw.split()
            if not w:
                continue
            if w[0] == "t":
                toks.append(int(w[1]))
            elif w[0] == "e":
                timings.append((int(w[1]), int(w[2]), float(w[3]), float(w[4])))
                break
            else:
                fail(f"unexpected runner line: {raw!r}")
        outputs.append(toks)
    p.stdin.close()
    p.wait()
    dumps = {}
    if dump:
        for fn in os.listdir(ddir):
            if fn.startswith("logits0_p"):
                pos = int(fn.split("_p")[1].split(".")[0])
                dumps[pos] = np.fromfile(os.path.join(ddir, fn), dtype=np.float32)
        shutil.rmtree(ddir)
    return outputs, timings, dumps


def run_runner_gates(args):
    from transformers import PreTrainedTokenizerFast

    test_gumbel_math()

    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.meta, "tokenizer.json"))
    tpl = open(os.path.join(args.meta, "chat_template.jinja")).read()

    # 2a. Short prompt: INK2_V1=1 (BF16 controller fix) + default vs pure FP32 ref over 16 decode steps
    stick_prompt = (
        "Create a stick fight game where player can fight a series of NPC enemies or PvP via couch co-op "
        "with up to 4 controllers. Outline the architecture, TypeScript modules, WebGPU renderer with WebGL2 "
        "fallback, and input handling."
    )
    ids_short = tok.encode(
        tok.apply_chat_template([{"role": "user", "content": stick_prompt}], chat_template=tpl, add_generation_prompt=True, tokenize=False),
        add_special_tokens=False,
    )
    ref_env = {
        "INK2_ONEROW": "1",
        "INK2_NOXMX": "1",
        "INK2_NODP4A": "1",
        "INK2_PRO": "1",
        "INK2_SYNC": "1",
        "INK2_OLDSHAPE": "1",
        "INK2_DOWN1": "1",
        "INK2_ROWS": "1",
        "INK2_NOPREFIX": "1",
    }
    res_v1, _, d_v1 = run_serve_batch(args.runner, args.model, 8192, {"INK2_V1": "1"}, [(ids_short, 8, 0.0, 1)], dump=True)
    res_ref, _, d_ref = run_serve_batch(args.runner, args.model, 8192, ref_env, [(ids_short, 16, 0.0, 1)], dump=True)
    res_def, _, d_def = run_serve_batch(args.runner, args.model, 8192, {"INK2_SYNC": "1"}, [(ids_short, 16, 0.0, 1)], dump=True)

    if res_v1[0] != res_ref[0][:8]:
        fail(f"INK2_V1 mismatch vs ref_fp32: {res_v1[0]} vs {res_ref[0][:8]}")
    ok(f"INK2_V1=1 healthy (BF16 controllers) and exact 8/8 match with ref_fp32")

    kls = [kl_div(d_ref[p], d_def[p]) for p in sorted(d_ref) if p in d_def]
    match16 = sum(a == b for a, b in zip(res_ref[0], res_def[0]))
    if match16 < 16 or max(kls) > 0.01 or np.mean(kls) > 0.002:
        fail(f"short decode parity failed: match={match16}/16 max_KL={max(kls):.5f} mean_KL={np.mean(kls):.5f}")
    ok(f"short 16-step decode parity: 16/16 greedy match, mean_KL={np.mean(kls):.5f}, max_KL={max(kls):.5f}")

    # 2b. Filtered sampling sanity (top_k=1 and top_p=0.01 match greedy; top_p=0.95 & min_p=0.05 work deterministically)
    res_filt, tim_filt, _ = run_serve_batch(
        args.runner,
        args.model,
        8192,
        {},
        [
            (ids_short, 16, 1.0, 1.0, 1, 0.0, 77),       # top_k=1 -> greedy
            (ids_short, 16, 1.0, 0.0001, 0, 0.0, 77),    # tiny top_p -> greedy
            (ids_short, 64, 1.0, 0.95, 40, 0.05, 77),    # filtered run A
            (ids_short, 64, 1.0, 0.95, 40, 0.05, 77),    # filtered run B (same seed -> identical)
        ],
    )
    if res_filt[0] != res_def[0]:
        fail(f"top_k=1 did not match greedy: {res_filt[0]} vs {res_def[0]}")
    if res_filt[1] != res_def[0]:
        fail(f"top_p=0.0001 did not match greedy: {res_filt[1]} vs {res_def[0]}")
    if res_filt[2] != res_filt[3]:
        fail("filtered sampling not deterministic for identical seed")
    filt_tps = (tim_filt[2][1] - 1) / tim_filt[2][3] if tim_filt[2][3] > 0 else 0
    ok(f"filtered sampling (top_k/top_p/min_p) exact, deterministic, {filt_tps:.1f} tok/s")

    # 2c. Long-context (> RING=1344) PrefixCkpt bit-exactness + 4096-token sampled coherence (including old crash seeds 1001/1002!)
    src_path = os.path.join(os.path.dirname(os.path.dirname(args.runner)), "inkling2_serve.py")
    if not os.path.exists(src_path):
        src_path = "/app/inkling2_serve.py"
    code_src = open(src_path).read()
    long_sys = (
        "You are Hermes Agent, an expert software engineering assistant.\n"
        "Reference server implementation:\n```python\n" + code_src[:7500] + "\n```"
    )
    m_t1 = [{"role": "system", "content": long_sys}, {"role": "user", "content": stick_prompt}]
    m_t2 = m_t1 + [
        {"role": "assistant", "content": "I will design the stick fight game in TypeScript with a WebGPU/WebGL2 renderer and 4-controller Gamepad API support."},
        {"role": "user", "content": "Now write the TypeScript GamepadManager class supporting up to 4 controllers with deadzones and combo buffering."},
    ]
    ids_t1 = tok.encode(tok.apply_chat_template(m_t1, chat_template=tpl, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
    ids_t2 = tok.encode(tok.apply_chat_template(m_t2, chat_template=tpl, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
    assert len(ids_t1) > 1400 and len(ids_t2) > len(ids_t1), f"expected > RING=1344, got {len(ids_t1)}, {len(ids_t2)}"

    res_cold, _, d_cold = run_serve_batch(args.runner, args.model, 16384, {"INK2_SYNC": "1", "INK2_NOPREFIX": "1"}, [(ids_t2, 8, 0.0, 1)], dump=True)
    res_cach, _, d_cach = run_serve_batch(args.runner, args.model, 16384, {"INK2_SYNC": "1"}, [(ids_t1, 32, 0.0, 1), (ids_t2, 8, 0.0, 1)], dump=True)
    if res_cold[0] != res_cach[1]:
        fail(f"PrefixCkpt mismatch at len={len(ids_t2)} > RING: {res_cold[0]} vs {res_cach[1]}")
    pos0 = len(ids_t2) - 1
    max_diff = max(float(np.max(np.abs(d_cold[p] - d_cach[p]))) for p in sorted(d_cold) if p >= pos0)
    if max_diff > 1e-6:
        fail(f"PrefixCkpt logit diff too high: {max_diff:.3e}")
    ok(f"PrefixCkpt at {len(ids_t2)} tok (> RING=1344) is BIT-EXACT cold vs cached (max_diff={max_diff:.2e}, 8/8 match)")

    # 2d. 4,096-token sampled coherence stress test (temp=1.0 top_p=1.0 AND top_p=0.95, including old +INF seeds 1001 & 1002!)
    reqs = [
        (ids_t1, 512, 1.0, 1.0, 0, 0.0, 1001),
        (ids_t2, 512, 1.0, 1.0, 0, 0.0, 1002),
        (ids_t1, 512, 1.0, 1.0, 0, 0.0, 2026),
        (ids_t2, 512, 1.0, 1.0, 0, 0.0, 2027),
        (ids_t1, 512, 1.0, 0.95, 0, 0.0, 3001),
        (ids_t2, 512, 1.0, 0.95, 0, 0.0, 3002),
        (ids_t1, 512, 1.0, 0.95, 40, 0.05, 4001),
        (ids_t2, 512, 1.0, 0.95, 40, 0.05, 4002),
    ]
    outs, tims, _ = run_serve_batch(args.runner, args.model, 16384, {}, reqs)
    total_toks = 0
    total_bad = 0
    for idx, (out, tim) in enumerate(zip(outs, tims)):
        total_toks += len(out)
        txt = tok.decode(out, skip_special_tokens=True)
        bad = find_stray_scripts(txt)
        tps = (tim[1] - 1) / tim[3] if tim[3] > 0 and tim[1] > 1 else 0.0
        print(f"  run {idx+1}/8: gen={len(out):3d} tok, prefill={tim[2]*1e3:6.1f} ms, decode={tps:5.1f} tok/s, stray={len(bad)}", flush=True)
        if bad:
            total_bad += len(bad)
            print(f"    STRAY CHARS: {bad[:10]} in {txt[:200]!r}", flush=True)
    if total_bad > 0:
        fail(f"detected {total_bad} stray non-Latin characters across {total_toks} sampled tokens!")
    ok(f"sampled coherence gate PASSED: 0 stray foreign-script characters across {total_toks} tokens (T=1.0, top_p=1.0 & 0.95, seeds 1001/1002 included)")


# ---------------------------------------------------------------------------- Gate 3: Live HTTP OpenAI + Hermes tool-calling + vision
def http_post(url, payload, timeout=180):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def run_http_gates(args):
    base = args.url.rstrip("/")
    # 3a. Health & models
    with urllib.request.urlopen(f"{base}/v1/models", timeout=15) as r:
        models = json.loads(r.read())
    model_id = models["data"][0]["id"]
    ok(f"GET /v1/models -> {model_id}")

    # 3b. Validation of bad sampling params returns HTTP 400
    for bad_param in ({"top_p": 0.0}, {"top_p": 1.5}, {"min_p": 1.0}, {"top_k": -5}):
        payload = {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        payload.update(bad_param)
        try:
            http_post(f"{base}/v1/chat/completions", payload)
            fail(f"expected HTTP 400 for {bad_param}")
        except urllib.error.HTTPError as e:
            if e.code != 400:
                fail(f"expected HTTP 400 for {bad_param}, got {e.code}")
    ok("invalid top_p/top_k/min_p cleanly rejected with HTTP 400")

    # 3c. Multi-turn Hermes-style tool-calling simulation with stick-fight prompt (T=1.0, top_p=1.0 AND top_p=0.95)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a shell command in the project workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Execute a Python snippet in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        },
    ]
    valid_tool_names = {"terminal", "execute_code"}
    sys_msg = (
        "You are Hermes Agent, an autonomous coding agent. Use the provided tools (terminal, execute_code) "
        "carefully with valid JSON arguments. Never invent extra tools."
    )
    user_msg = (
        "Create a stick fight game where a player can fight a series of NPC enemies or PvP via couch co-op "
        "with up to 4 controllers. First inspect the current directory using the terminal tool."
    )
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

    # Turn 1: expect clean tool call
    _, raw1 = http_post(
        f"{base}/v1/chat/completions",
        {
            "model": model_id,
            "messages": messages,
            "tools": tools,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 1001,
            "max_tokens": 512,
        },
    )
    r1 = json.loads(raw1)
    m1 = r1["choices"][0]["message"]
    t1_txt = (m1.get("reasoning_content") or "") + "\n" + (m1.get("content") or "")
    bad1 = find_stray_scripts(t1_txt)
    if bad1:
        fail(f"Turn 1 stray chars: {bad1[:10]} in {t1_txt[:300]!r}")
    tcs = m1.get("tool_calls") or []
    if not tcs:
        fail(f"Turn 1 expected tool_call, got none: {m1}")
    for tc in tcs:
        fn = tc["function"]
        if fn["name"] not in valid_tool_names:
            fail(f"Turn 1 invalid/corrupted tool name: {fn['name']!r}")
        parsed = json.loads(fn["arguments"])
        if not isinstance(parsed, dict) or "_raw" in parsed:
            fail(f"Turn 1 malformed tool JSON args: {fn['arguments']!r}")
        if find_stray_scripts(fn["arguments"]):
            fail(f"Turn 1 stray chars in tool args: {fn['arguments']!r}")
    ok(
        f"Hermes Turn 1 tool_call clean: name={tcs[0]['function']['name']!r} args={tcs[0]['function']['arguments']} "
        f"(decode={r1['timings']['predicted_per_second']:.1f} tok/s)"
    )

    # Turn 2: feed tool result back, stream response with top_p=0.95 & 768 tokens
    messages.append(m1)
    messages.append({"role": "tool", "tool_call_id": tcs[0]["id"], "name": tcs[0]["function"]["name"], "content": "total 0\n(empty workspace directory)"})
    messages.append({"role": "user", "content": "Great, now reply with the complete architecture plan and TypeScript interface definitions (no tool call needed)."})
    req2 = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model_id,
                "messages": messages,
                "tools": tools,
                "reasoning_effort": "low",
                "temperature": 1.0,
                "top_p": 0.95,
                "min_p": 0.02,
                "seed": 1002,
                "max_tokens": 1024,
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    stream_r, stream_c = [], []
    tim2 = None
    with urllib.request.urlopen(req2, timeout=180) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            d = chunk["choices"][0]["delta"]
            if d.get("reasoning_content"):
                stream_r.append(d["reasoning_content"])
            if d.get("content"):
                stream_c.append(d["content"])
            if chunk.get("timings"):
                tim2 = chunk["timings"]
    full2 = "".join(stream_r) + "\n" + "".join(stream_c)
    bad2 = find_stray_scripts(full2)
    if bad2:
        fail(f"Turn 2 streaming stray chars: {bad2[:10]}")
    if len("".join(stream_c)) < 100:
        fail(f"Turn 2 streaming answer unexpectedly short: {''.join(stream_c)!r}")
    ok(
        f"Hermes Turn 2 SSE streaming (top_p=0.95, min_p=0.02) clean: {len(full2)} chars, 0 stray chars, "
        f"prefill={tim2['prompt_ms']:.1f} ms, decode={tim2['predicted_per_second']:.1f} tok/s"
    )

    # 3d. Vision regression (synthetic high-contrast image + Turn-2 cache hit)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        ok("skipping vision HTTP check (PIL not installed on host; run inside container to include vision)")
        return

    img = Image.new("RGB", (200, 120), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.rectangle([20, 20, 180, 100], fill=(220, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    v_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "What color is the rectangle in this image? Reply in one word."},
            ],
        }
    ]
    _, rv1 = http_post(f"{base}/v1/chat/completions", {"model": model_id, "messages": v_msgs, "temperature": 0.0, "max_tokens": 128})
    jv1 = json.loads(rv1)
    ans_v1 = jv1["choices"][0]["message"]["content"].lower()
    if "red" not in ans_v1:
        fail(f"vision color check failed: got {ans_v1!r}")
    _, rv2 = http_post(f"{base}/v1/chat/completions", {"model": model_id, "messages": v_msgs, "temperature": 0.0, "max_tokens": 64})
    jv2 = json.loads(rv2)
    if jv2["timings"].get("vision_ms", 1.0) != 0.0:
        fail(f"vision image cache did not hit on Turn 2: {jv2['timings']}")
    ok(f"vision gate PASSED: answer={ans_v1.strip()!r}, T1 vision_ms={jv1['timings']['vision_ms']:.1f}, T2 cached vision_ms=0.0")

    # 3e. Streaming client-disconnect abort recovery (regression for BrokenPipeError pipe-desync bug)
    parsed = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=60)
    abort_payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "Count slowly from 1 to 80 separated by commas."}],
        "reasoning_effort": "low",
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": True,
    }).encode()
    conn.request("POST", "/v1/chat/completions", body=abort_payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        fail(f"disconnect test expected HTTP 200, got {resp.status}")
    seen_chunks = 0
    while seen_chunks < 2:
        line = resp.readline()
        if not line:
            break
        if line.startswith(b"data: "):
            seen_chunks += 1
    # Abruptly drop socket mid-stream before server finishes generating!
    conn.close()

    # Follow-up request MUST succeed cleanly with zero pipe desync (also verifies reasoning_effort="off" -> "none")
    _, post_raw = http_post(
        f"{base}/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "What is 17*23? Reply with only the number."}],
            "reasoning_effort": "off",
            "temperature": 0.0,
            "max_tokens": 64,
        },
        timeout=60,
    )
    post_j = json.loads(post_raw)
    post_ans = (post_j["choices"][0]["message"].get("content") or "").strip()
    if "391" not in post_ans:
        fail(f"post-disconnect follow-up failed or desynced: got {post_ans!r}")
    ok(f"streaming disconnect recovery + reasoning_effort='off' PASSED: follow-up answer={post_ans!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("runner", "http"), required=True)
    ap.add_argument("--runner", default="/app/bin/xpu-inkling2")
    ap.add_argument("--model", default="/models/Inkling-2-Mini-UD-Q4_K_XL.gguf")
    ap.add_argument("--meta", default="/app/meta")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    if args.mode == "runner":
        run_runner_gates(args)
    else:
        run_http_gates(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark three Llama-3-8B variants: BF16, SCLP-compact, Q8_0.

Runs:
  1. Perplexity     — llama-perplexity on WikiText-2
  2. Speed          — llama-bench (prompt processing + token generation)
  3. Task accuracy  — lm-evaluation-harness via llama-server
     Tasks: hellaswag, arc_challenge, arc_easy, winogrande, mmlu

Usage:
    source eval_env/bin/activate
    python3 tests/benchmark_all.py [--models bf16 sclp q8] [--skip-lmeval] [--output results.md]

Environment variables:
    LLAMA_CPP_BIN   path to llama.cpp build/bin/   (default: ../llama.cpp/build/bin)
    WIKITEXT        path to WikiText-2 text file    (default: /tmp/wikitext2.txt)
    MODEL_DIR       path to model directory         (default: models/llama3)
"""

import argparse
import concurrent.futures
import csv
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import signal
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIN  = Path(os.environ.get('LLAMA_CPP_BIN', REPO.parent / 'llama.cpp' / 'build' / 'bin'))
WIKITEXT  = Path(os.environ.get('WIKITEXT',  '/tmp/wikitext2.txt'))
MODEL_DIR = Path(os.environ.get('MODEL_DIR', REPO / 'models' / 'llama3'))

MODELS = {
    'bf16': {
        'label': 'BF16 baseline',
        'file':  'Meta-Llama-3-8B.fp16.gguf',
        'size_gb': 15.0,
    },
    'sclp': {
        'label': 'SCLP compact',
        'file':  'Llama-3-8B-SCLP-Compact.gguf',
        'size_gb': 11.72,
    },
    'q8': {
        'label': 'Q8_0',
        'file':  'Llama-3-8B-Q8_0.gguf',
        'size_gb': 8.0,
    },
}

LM_EVAL_TASKS = ['hellaswag', 'arc_challenge', 'arc_easy']
SERVER_PORT   = 8765
PROXY_PORT    = 8766  # converts llama-server logprobs format → OpenAI token_logprobs format
N_GPU_LAYERS  = 99
CTX_SIZE      = 16384  # 16 parallel slots × 1024 tokens/slot
N_PARALLEL    = 16


# ── helpers ───────────────────────────────────────────────────────────────────

def run(cmd, timeout=600, check=True):
    cmd = [str(c) for c in cmd]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout + result.stderr


def ensure_wikitext():
    if WIKITEXT.exists():
        return
    print(f"Downloading WikiText-2 to {WIKITEXT}...")
    import urllib.request
    url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
    # Prefer the real WikiText-2 test split if available via datasets
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n'.join(ds['text'])
        WIKITEXT.write_text(text)
        print(f"  Saved {len(text):,} chars from HuggingFace datasets.")
    except Exception:
        urllib.request.urlretrieve(url, WIKITEXT)
        print(f"  Saved PTB test set as fallback.")


# ── perplexity ────────────────────────────────────────────────────────────────

def run_ppl(model_path: Path) -> dict:
    cmd = [
        BIN / 'llama-perplexity',
        '-m', model_path,
        '-f', WIKITEXT,
        '-ngl', N_GPU_LAYERS,
        '--ctx-size', '512',  # standard WikiText-2 PPL window
    ]
    print(f"  Running PPL... ", end='', flush=True)
    t0 = time.time()
    out = run(cmd, timeout=1800)
    elapsed = time.time() - t0

    m = re.search(r'Final estimate.*?PPL\s*=\s*([\d.]+)', out)
    if not m:
        matches = re.findall(r'PPL\s*=\s*([\d.]+)', out)
        ppl = float(matches[-1]) if matches else float('nan')
    else:
        ppl = float(m.group(1))
    print(f"PPL={ppl:.4f} ({elapsed:.0f}s)")
    return {'ppl': ppl}


# ── speed (llama-bench) ───────────────────────────────────────────────────────

def run_bench(model_path: Path) -> dict:
    cmd = [
        BIN / 'llama-bench',
        '-m', model_path,
        '-ngl', N_GPU_LAYERS,
        '-p', '512',   # prompt tokens
        '-n', '128',   # generation tokens
        '-r', '3',     # repetitions
        '--output', 'csv',
    ]
    print(f"  Running llama-bench... ", end='', flush=True)
    out = run(cmd, timeout=600)

    # CSV output has quoted fields; avg_ts is col 39, n_prompt col 33, n_gen col 34.
    pp_vals, tg_vals = [], []
    reader = csv.reader(io.StringIO(out))
    header = None
    for row in reader:
        if not row:
            continue
        if header is None and row[0].strip().startswith('build_commit'):
            header = row
            continue
        if header is None or len(row) < 40:
            continue
        try:
            n_prompt = int(row[33])
            n_gen    = int(row[34])
            avg_ts   = float(row[39])
            if n_prompt > 0 and n_gen == 0:
                pp_vals.append(avg_ts)
            elif n_gen > 0 and n_prompt == 0:
                tg_vals.append(avg_ts)
        except (ValueError, IndexError):
            pass

    pp = sum(pp_vals) / len(pp_vals) if pp_vals else float('nan')
    tg = sum(tg_vals) / len(tg_vals) if tg_vals else float('nan')
    print(f"pp={pp:.1f} t/s  tg={tg:.1f} t/s")
    return {'pp_tps': pp, 'tg_tps': tg}


# ── lm-eval via llama-server ──────────────────────────────────────────────────

def start_server(model_path: Path):
    cmd = [
        BIN / 'llama-server',
        '-m', model_path,
        '-ngl', N_GPU_LAYERS,
        '--ctx-size', CTX_SIZE,
        '--port', SERVER_PORT,
        '--host', '127.0.0.1',
        '-np', str(N_PARALLEL),
        '--log-disable',
    ]
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    import urllib.request
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{SERVER_PORT}/health', timeout=2)
            return proc
        except Exception:
            pass
    proc.kill()
    raise RuntimeError("llama-server failed to start within 120s")


def stop_server(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def _backend_post(backend_port: int, path: str, body: bytes, timeout: int = 120) -> dict:
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(
            f'http://127.0.0.1:{backend_port}{path}',
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        ),
        timeout=timeout,
    ).read())


def _compute_echo_logprobs(prompt, backend_port: int):
    """Compute per-token logprobs for the full prompt by making parallel prefix requests.

    llama-server silently ignores echo=True and returns only the generated token logprob.
    lm-eval needs token_logprobs[ctxlen:-1] (continuation slice) which would be empty
    with only 1 entry. Fix: compute P(token[i] | token[0..i-1]) for all positions via
    parallel one-step requests, return the full token_logprobs array.
    """
    # Tokenize the full prompt text (or pass token IDs directly)
    if isinstance(prompt, str):
        tok_body = json.dumps({'content': prompt}).encode()
    else:
        tok_body = json.dumps({'content': prompt, 'add_special': False}).encode()
    try:
        tok_resp = _backend_post(backend_port, '/tokenize', tok_body, timeout=30)
        token_ids = tok_resp['tokens']
    except Exception:
        return None  # fall back to passthrough

    N = len(token_ids)
    if N == 0:
        return None

    def score_position(i):
        """Return logprob of token_ids[i] given prefix token_ids[0..i-1]."""
        req = json.dumps({
            'prompt': token_ids[:i],
            'max_tokens': 1,
            'logprobs': True,
            'top_logprobs': 100,
            'temperature': 0,
        }).encode()
        try:
            resp = _backend_post(backend_port, '/v1/completions', req, timeout=120)
            content_entry = resp['choices'][0]['logprobs']['content'][0]
            target = token_ids[i]
            for entry in content_entry.get('top_logprobs', []):
                if entry['id'] == target:
                    return entry['logprob'], entry['token']
            # Target not in top-100: use min logprob minus a conservative penalty
            tops = content_entry.get('top_logprobs', [])
            if tops:
                min_lp = min(e['logprob'] for e in tops)
                return min_lp - 4.0, None
            return -100.0, None
        except Exception:
            return -100.0, None

    token_logprobs = [None]  # position 0 has no prior context
    token_texts = [None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
        futs = {ex.submit(score_position, i): i for i in range(1, N)}
        pos_results = {}
        for f in concurrent.futures.as_completed(futs):
            pos_results[futs[f]] = f.result()

    for i in range(1, N):
        lp, tok = pos_results[i]
        token_logprobs.append(lp)
        token_texts.append(tok)

    return token_logprobs, token_texts


def start_logprob_proxy(backend_port: int, proxy_port: int) -> HTTPServer:
    """Proxy that implements echo=True properly for lm-eval loglikelihood scoring.

    llama-server ignores echo=True. We compute per-token logprobs via parallel
    prefix requests so lm-eval's token_logprobs[ctxlen:-1] slice works correctly.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                req = json.loads(body)
            except Exception:
                req = {}

            try:
                if req.get('echo'):
                    # lm-eval loglikelihood: needs per-token logprobs for full prompt.
                    # Build a proper echo response without relying on llama-server echo.
                    prompts = req.get('prompt', '')
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    elif not isinstance(prompts, list):
                        prompts = [prompts]

                    # For each prompt in the batch, compute echo logprobs in parallel.
                    def compute_one(p):
                        result = _compute_echo_logprobs(p, backend_port)
                        return result

                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
                        echo_results = list(ex.map(compute_one, prompts))

                    # Also call backend once (without echo) to get the generated token info.
                    no_echo_req = dict(req)
                    no_echo_req.pop('echo', None)
                    no_echo_body = json.dumps(no_echo_req).encode()
                    backend_resp = _backend_post(backend_port, self.path, no_echo_body)

                    # Merge: extend token_logprobs with generated token logprob.
                    choices = backend_resp.get('choices', [])
                    for idx, choice in enumerate(sorted(choices, key=lambda c: c.get('index', 0))):
                        echo_res = echo_results[idx] if idx < len(echo_results) else None
                        if echo_res is None:
                            continue
                        token_logprobs, token_texts = echo_res

                        # Generated token's logprob from backend
                        gen_lp_list = choice.get('logprobs') or {}
                        gen_content = gen_lp_list.get('content', [])
                        gen_lp = gen_content[0]['logprob'] if gen_content else None
                        gen_tok = gen_content[0]['token'] if gen_content else None

                        full_lps = token_logprobs + [gen_lp]
                        full_toks = token_texts + [gen_tok]
                        # Use a fallback key for unknown tokens so max(top.values()) won't crash
                        top_lps = [{t if t else '<unk>': lp if lp is not None else -100.0}
                                   for t, lp in zip(full_toks, full_lps)]

                        choice['logprobs'] = {
                            'token_logprobs': full_lps,
                            'tokens': full_toks,
                            'top_logprobs': top_lps,
                        }

                    out = json.dumps(backend_resp).encode()
                else:
                    # Non-echo request: pass through, converting content[].logprob format.
                    data = _backend_post(backend_port, self.path, body)
                    for choice in data.get('choices', []):
                        lp = choice.get('logprobs') or {}
                        if 'content' in lp and 'token_logprobs' not in lp:
                            choice['logprobs']['token_logprobs'] = [c['logprob'] for c in lp['content']]
                            choice['logprobs']['tokens'] = [c['token'] for c in lp['content']]
                            choice['logprobs']['top_logprobs'] = [{c['token']: c['logprob']} for c in lp['content']]
                    out = json.dumps(data).encode()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(out))
                self.end_headers()
                self.wfile.write(out)
            except Exception as e:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(str(e).encode())

        def do_GET(self):
            try:
                data = urllib.request.urlopen(
                    f'http://127.0.0.1:{backend_port}{self.path}', timeout=10
                ).read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(502)
                self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', proxy_port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def run_lmeval(model_path: Path, limit: int = 1000) -> dict:
    print(f"  Starting llama-server...", end='', flush=True)
    proc = start_server(model_path)
    print(" ready.")

    proxy = start_logprob_proxy(SERVER_PORT, PROXY_PORT)
    tokenizer = str(REPO.parent / 'llama.cpp' / 'gguf-py')  # fallback; override with HF path if available
    hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
    llama3_snaps = sorted((hf_cache / 'models--unsloth--llama-3-8b' / 'snapshots').glob('*'))
    if llama3_snaps:
        tokenizer = str(llama3_snaps[-1])
    base_url = f'http://127.0.0.1:{PROXY_PORT}/v1/completions'
    cmd = [
        sys.executable, '-m', 'lm_eval',
        '--model', 'local-completions',
        '--model_args', f'model={tokenizer},base_url={base_url},num_concurrent={N_PARALLEL},max_retries=3,tokenized_requests=False',
        '--tasks', ','.join(LM_EVAL_TASKS),
        '--num_fewshot', '0',
        '--batch_size', 'auto',
        '--limit', str(limit),
        '--output_path', '/tmp/lmeval_results',
    ]
    scores = {}
    output_dir = Path('/tmp/lmeval_results')
    try:
        print(f"  Running lm-eval tasks ({', '.join(LM_EVAL_TASKS)})...")
        # Clear old results so we pick up only this run's output.
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        lmeval_out = run(cmd, timeout=18000, check=False)

        # Read scores from the JSON output file written by lm_eval.
        result_files = sorted(output_dir.rglob('results_*.json'))
        if result_files:
            results_json = json.loads(result_files[-1].read_text())
            for task in LM_EVAL_TASKS:
                task_res = results_json.get('results', {}).get(task, {})
                # Prefer acc_norm if available, else acc.
                val = task_res.get('acc_norm,none', task_res.get('acc,none', float('nan')))
                scores[task] = float(val)
        else:
            print(f"  WARNING: lm_eval produced no JSON results.\n  --- HEAD ---\n{lmeval_out[:2000]}\n  --- TAIL ---\n{lmeval_out[-1000:]}")
            for task in LM_EVAL_TASKS:
                scores[task] = float('nan')
    finally:
        proxy.shutdown()
        stop_server(proc)

    return scores


# ── formatting ────────────────────────────────────────────────────────────────

def fmt(val, fmt_str='.2f', missing='—'):
    if val != val:  # nan
        return missing
    return format(val, fmt_str)


def build_report(results: dict) -> str:
    lines = ['# Benchmark Results: Llama-3-8B Variants\n']
    lines.append(f'*Hardware: AMD RX 7900 XTX (gfx1100) · GGML_HIP=ON · ngl=99 · ctx={CTX_SIZE}*\n')

    # File sizes
    lines.append('## File Size\n')
    lines.append('| Variant | File size |')
    lines.append('|---|---|')
    for key, info in MODELS.items():
        if key in results:
            lines.append(f"| {info['label']} | {info['size_gb']:.2f} GB |")
    lines.append('')

    # PPL
    lines.append('## Perplexity (WikiText-2)\n')
    lines.append('| Variant | PPL |')
    lines.append('|---|---|')
    for key, info in MODELS.items():
        if key in results and 'ppl' in results[key]:
            lines.append(f"| {info['label']} | {fmt(results[key]['ppl'], '.4f')} |")
    lines.append('')

    # Speed
    lines.append('## Speed (llama-bench, 512 prompt + 128 generation tokens)\n')
    lines.append('| Variant | Prompt (t/s) | Generation (t/s) |')
    lines.append('|---|---|---|')
    for key, info in MODELS.items():
        if key in results and 'pp_tps' in results[key]:
            r = results[key]
            lines.append(f"| {info['label']} | {fmt(r['pp_tps'], '.1f')} | {fmt(r['tg_tps'], '.1f')} |")
    lines.append('')

    # Task accuracy
    if any('hellaswag' in results.get(k, {}) for k in results):
        lines.append('## Task Accuracy (0-shot, lm-evaluation-harness)\n')
        header = '| Variant | HellaSwag | ARC-C | ARC-E | WinoGrande | MMLU |'
        lines.append(header)
        lines.append('|---|---|---|---|---|---|')
        for key, info in MODELS.items():
            if key in results:
                r = results[key]
                row = (f"| {info['label']} "
                       f"| {fmt(r.get('hellaswag', float('nan')), '.3f')} "
                       f"| {fmt(r.get('arc_challenge', float('nan')), '.3f')} "
                       f"| {fmt(r.get('arc_easy', float('nan')), '.3f')} "
                       f"| {fmt(r.get('winogrande', float('nan')), '.3f')} "
                       f"| {fmt(r.get('mmlu', float('nan')), '.3f')} |")
                lines.append(row)
        lines.append('')

    return '\n'.join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark BF16, SCLP, and Q8_0 Llama-3-8B variants.")
    parser.add_argument('--models', nargs='+', choices=['bf16', 'sclp', 'q8'], default=['bf16', 'sclp', 'q8'])
    parser.add_argument('--skip-ppl',    action='store_true', help='Skip perplexity benchmark')
    parser.add_argument('--skip-bench',  action='store_true', help='Skip llama-bench speed test')
    parser.add_argument('--skip-lmeval', action='store_true', help='Skip lm-evaluation-harness tasks')
    parser.add_argument('--limit', type=int, default=1000, help='Max examples per task for lm-eval')
    parser.add_argument('--output', default=None, help='Write markdown report to this file')
    args = parser.parse_args()

    ensure_wikitext()

    results = {}
    for key in args.models:
        info = MODELS[key]
        model_path = MODEL_DIR / info['file']
        if not model_path.exists():
            print(f"SKIP {info['label']}: {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {info['label']}  ({model_path.name})")
        print(f"{'='*60}")
        results[key] = {}

        if not args.skip_ppl:
            results[key].update(run_ppl(model_path))
        if not args.skip_bench:
            results[key].update(run_bench(model_path))
        if not args.skip_lmeval:
            results[key].update(run_lmeval(model_path, limit=args.limit))

    report = build_report(results)
    print(f"\n\n{report}")

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        out = REPO / 'benchmark_results.md'
        out.write_text(report)
        print(f"Report written to {out}")


if __name__ == '__main__':
    main()

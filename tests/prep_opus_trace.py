#!/usr/bin/env python3
"""
Convert Verdugie/opus-4.6-training-catalog JSONL into plain-text files
for llama-imatrix calibration and llama-perplexity OOD eval.

Format: each conversation flattened as "USER: ...\n\nASSISTANT: ...\n\n"
between conversations. System messages are inlined as "[SYSTEM]: ...".

Held-out tail of each split goes to ood_eval.txt; the rest is calibration text.
"""
import json
import os
import argparse
import random

ROLE_PREFIX = {'system': '[SYSTEM]', 'user': 'USER', 'assistant': 'ASSISTANT'}


def _content_to_text(content):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                if 'text' in block:
                    out.append(block['text'])
                elif 'content' in block:
                    out.append(_content_to_text(block['content']))
            elif isinstance(block, str):
                out.append(block)
        return "\n".join(out)
    return str(content)


def flatten(messages):
    parts = []
    for m in messages:
        role = ROLE_PREFIX.get(m['role'], m['role'].upper())
        content = _content_to_text(m.get('content')).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/home/ajkerchum/poc/eval_data/opus-trace')
    ap.add_argument('--out-dir',  default='/home/ajkerchum/poc/eval_data/opus-trace')
    ap.add_argument('--holdout-per-split', type=int, default=50,
                    help='How many conversations from each split to hold out for OOD eval')
    ap.add_argument('--max-cal-conversations', type=int, default=5000,
                    help='Cap calibration conversations (reasoning.jsonl has 166k)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    cal_chunks = []
    ood_chunks = []

    for fname in ['conversation.jsonl', 'coding.jsonl', 'reasoning.jsonl']:
        path = os.path.join(args.data_dir, fname)
        with open(path) as fp:
            convos = [json.loads(line) for line in fp]
        random.shuffle(convos)
        holdout = convos[:args.holdout_per_split]
        rest    = convos[args.holdout_per_split:]
        # Cap reasoning since it's massive
        rest = rest[:args.max_cal_conversations]
        print(f"{fname}: total={len(convos)}, holdout={len(holdout)}, calibration={len(rest)}")

        for c in holdout:
            ood_chunks.append(flatten(c['messages']))
        for c in rest:
            cal_chunks.append(flatten(c['messages']))

    cal_path = os.path.join(args.out_dir, 'imatrix_cal.txt')
    ood_path = os.path.join(args.out_dir, 'ood_eval.txt')

    with open(cal_path, 'w') as f:
        f.write("\n\n=====\n\n".join(cal_chunks))
    with open(ood_path, 'w') as f:
        f.write("\n\n=====\n\n".join(ood_chunks))

    print(f"\nWrote: {cal_path} ({os.path.getsize(cal_path)/1024/1024:.1f} MB, {len(cal_chunks)} convos)")
    print(f"Wrote: {ood_path} ({os.path.getsize(ood_path)/1024/1024:.1f} MB, {len(ood_chunks)} convos)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Capture greedy completions for a fixed prompt set, to a JSON file.

Speculative decoding is supposed to be LOSSLESS: spec-on and spec-off must
produce the same greedy tokens. Run this against each arm and diff the files.

Two things this is careful about, both from SESSION-HANDOFF.md §3:

* `--repeat N` runs every prompt N times against the SAME server, which is the
  determinism control. Greedy is not reproducible at long context on either
  arm, so if a prompt disagrees with itself here, it cannot be used to compare
  arms -- run the control before trusting any cross-arm diff.
* Prompts are deliberately SHORT. The non-determinism sets in from ~15k tokens.

  python3 greedy_capture.py --out spec_on.json --repeat 2
  python3 greedy_capture.py --out spec_off.json --repeat 2
  python3 greedy_capture.py --compare spec_on.json spec_off.json
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Q: What is 17 * 24? Think step by step, then give the answer.\nA:",
    "List the first 12 prime numbers, separated by commas.",
    "Write a Python function that reverses a linked list. Explain it after.",
    "Q: A train travels 60 km in 45 minutes. What is its speed in km/h?\nA:",
    "Explain what a KV cache is in transformer inference, in one paragraph.",
    "Translate to French, then back to English: 'The quick brown fox jumps.'",
    "Q: If x + 2y = 10 and x - y = 1, solve for x and y.\nA:",
    "Name the planets of the solar system in order from the Sun.",
]


def complete(base_url, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["choices"][0]["text"]


def do_compare(path_a, path_b):
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    same = diff = 0
    for i, prompt in enumerate(PROMPTS):
        key = str(i)
        if key not in a or key not in b:
            continue
        # Compare the first run of each arm.
        ta, tb = a[key][0], b[key][0]
        if ta == tb:
            same += 1
        else:
            diff += 1
            print(f"\n=== MISMATCH on prompt {i}: {prompt[:60]!r}")
            for j, (ca, cb) in enumerate(zip(ta, tb)):
                if ca != cb:
                    print(f"  diverges at char {j}")
                    print(f"    {path_a}: ...{ta[max(0, j - 40):j + 60]!r}")
                    print(f"    {path_b}: ...{tb[max(0, j - 40):j + 60]!r}")
                    break
            else:
                print(f"  one is a prefix of the other ({len(ta)} vs {len(tb)} chars)")
    print(f"\n{same} identical, {diff} different (of {same + diff} compared)")
    return 0 if diff == 0 else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="/data/DeepSeek-V4-Pro-0813")
    p.add_argument("--out")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()

    if args.compare:
        sys.exit(do_compare(*args.compare))
    if not args.out:
        p.error("--out is required unless --compare is used")

    results: dict[str, list[str]] = {}
    unstable = []
    for i, prompt in enumerate(PROMPTS):
        runs = [
            complete(args.base_url, args.model, prompt, args.max_tokens, args.timeout)
            for _ in range(args.repeat)
        ]
        results[str(i)] = runs
        if len(set(runs)) != 1:
            unstable.append(i)
        print(f"prompt {i}: {'STABLE' if len(set(runs)) == 1 else 'UNSTABLE'}")

    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    if unstable:
        print(
            f"DETERMINISM CONTROL FAILED for prompts {unstable}: the server "
            "disagrees with ITSELF, so these cannot be used to compare arms."
        )
    else:
        print("determinism control passed: every prompt reproduced itself.")


if __name__ == "__main__":
    main()

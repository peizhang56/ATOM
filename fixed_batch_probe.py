#!/usr/bin/env python3
"""Hold the running batch EXACTLY constant for a long decode.

`accept_probe.py` lets requests stop on EOS, so the running batch drains
(32 -> 26 -> 22 -> 0) over a long run. That matters here: under FULL cudagraphs
vLLM re-dispatches to a different capture bucket whenever the batch changes, so
a draining batch cannot distinguish "acceptance decays with decode position"
from "acceptance breaks when the capture bucket changes".

`ignore_eos` + identical `max_tokens` makes every request run the same number of
steps and finish together, so the batch is one number for the whole run and the
bucket never changes. Compare against `accept_probe.py` at the same
concurrency: if acceptance holds here and collapses there, the decode position
is innocent and the bucket transition is the defect.

  python3 fixed_batch_probe.py --concurrency 32 --max-tokens 2048
"""

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

PROMPT = (
    "You are a careful technical writer. Explain, in detail and step by step, "
    "how a speculative decoding system verifies a block of draft tokens "
    "against a target model, why rejection sampling preserves the target's "
    "output distribution, and what happens to the KV cache when a draft token "
    "is rejected. Be thorough and precise.\n\n"
)


def one(args, i):
    body = json.dumps(
        {
            "model": args.model,
            "prompt": PROMPT + f"Request {i}.",
            "max_tokens": args.max_tokens,
            "min_tokens": args.max_tokens,  # no early stop
            "ignore_eos": True,  # ...and none on EOS either
            "temperature": 0.0,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{args.base_url}/v1/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        return json.load(r)["usage"]["completion_tokens"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="/data/DeepSeek-V4-Pro-0813")
    p.add_argument("--timeout", type=float, default=3600)
    args = p.parse_args()

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        counts = list(ex.map(lambda i: one(args, i), range(args.concurrency)))
    dt = time.time() - t0
    print(
        f"{len(counts)}/{args.concurrency} requests, {sum(counts)} output tokens "
        f"in {dt:.1f}s"
    )
    # The whole point: every request must have run the same number of steps.
    print(f"  per-request completion tokens: min={min(counts)} max={max(counts)}")
    if min(counts) != max(counts):
        print("  WARNING: lengths differ, so the batch was NOT constant.")


if __name__ == "__main__":
    main()

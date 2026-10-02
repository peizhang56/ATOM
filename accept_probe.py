#!/usr/bin/env python3
"""Drive enough concurrent decode to make the server log a SpecDecoding line.

Acceptance is what this measures, not throughput, so the prompts are cheap and
the output is long: the metric is per decode STEP, and a step only exists once
every request is past prefill. The server logs
``SpecDecoding metrics: Mean acceptance length: ...`` on its own cadence; this
just keeps the batch full long enough for several of those to land.

  python3 accept_probe.py --concurrency 32 --max-tokens 256
"""

import argparse
import asyncio
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


def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


async def one(sem, args, i, stats):
    async with sem:
        payload = {
            "model": args.model,
            # Distinct prefixes so prefix caching cannot collapse the batch into
            # one shared trace -- acceptance would then not be representative.
            "prompt": f"[request {i}] " + PROMPT,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        t0 = time.time()
        try:
            out = await asyncio.to_thread(
                post, f"{args.base_url}/v1/completions", payload, args.timeout
            )
            n = out["usage"]["completion_tokens"]
            stats.append((n, time.time() - t0))
        except Exception as exc:  # noqa: BLE001
            print(f"  request {i} failed: {exc}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="/data/DeepSeek-V4-Pro-0813")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--requests", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=1800.0)
    args = p.parse_args()
    n_req = args.requests or args.concurrency

    sem = asyncio.Semaphore(args.concurrency)
    stats: list[tuple[int, float]] = []
    t0 = time.time()
    await asyncio.gather(*(one(sem, args, i, stats) for i in range(n_req)))
    dt = time.time() - t0

    done = len(stats)
    toks = sum(n for n, _ in stats)
    print(f"\n{done}/{n_req} requests, {toks} output tokens in {dt:.1f}s")
    if done:
        print(f"  output throughput: {toks / dt:.1f} tok/s")
    print("\nRead acceptance from the SERVER log:")
    print("  grep 'SpecDecoding metrics' logs/<server log> | tail -5")


if __name__ == "__main__":
    asyncio.run(main())

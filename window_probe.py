"""Long prompt, few vs many output tokens.

With a >=128-token prompt, the draft's 128-row window is initially ALL
prefill-written rows; decode-written rows displace them as generation runs.
"""
import json, sys, urllib.request, concurrent.futures as cf
MAXTOK = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE = ("Paris is the capital of France. Berlin is the capital of Germany. "
        "Rome is the capital of Italy. Madrid is the capital of Spain. ") * 12
def one(i):
    body = json.dumps({"model": "/data/DeepSeek-V4-Pro-0813",
                       "prompt": BASE + f" Question {i}: list five European capitals.",
                       "max_tokens": MAXTOK, "temperature": 0, "seed": 0}).encode()
    r = urllib.request.Request("http://localhost:8000/v1/completions", body,
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f:
        return json.load(f)["usage"]["completion_tokens"]
with cf.ThreadPoolExecutor(N) as ex:
    tot = sum(ex.map(one, range(N)))
print(f"max_tokens={MAXTOK} n={N} completion_tokens={tot}")

"""Send ONE fixed prompt, greedy, so both arms dump the same prefill."""
import sys, json, urllib.request
PROMPT = ("The capital of France is Paris. The capital of Germany is Berlin. "
          "The capital of Italy is")
body = json.dumps({
    "model": "/data/DeepSeek-V4-Pro-0813",
    "prompt": PROMPT,
    "max_tokens": 16,
    "temperature": 0,
    "seed": 0,
}).encode()
req = urllib.request.Request("http://localhost:8000/v1/completions", body,
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.load(r)
print("completion:", repr(out["choices"][0]["text"]))

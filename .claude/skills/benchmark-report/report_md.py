#!/usr/bin/env python3
"""Render a run_gsm8k_benchmark.sh results dir as a short standalone report.md.

Usage:
    report_md.py RESULTS_DIR [-o OUT] [--gpus N] [--title T]
                 [--server-script PATH] [--client-script PATH] [--note TEXT]

Autodetects serve_*.sh / client_*.sh in the project root (RESULTS_DIR/../..).
"""
from __future__ import annotations

import argparse, glob, json, math, os, re, sys

NS = 1e9


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def counter(metrics, name):
    m = metrics.get(name)
    if not m:
        return None
    tot, seen = 0.0, False
    for s in m.get("series", []):
        v = s.get("stats", {}).get("total")
        if v is not None:
            tot, seen = tot + v, True
    return tot if seen else None


def spec_stats(mm):
    """Speculative-decoding acceptance for one point, or None if absent.

    vLLM only registers vllm:spec_decode_* when --speculative-config is set, so
    absence means the server ran without spec decode -- not a lost measurement.
    aiperf strips the _total suffix, matching the prefix_cache keys above.

    Two ratios, because they answer different questions:
      accept% = accepted / drafted -- how often the draft head is right, the
                number to compare against num_speculative_tokens.
      tok/step = 1 + accepted / drafts -- tokens emitted per verify pass. This
                is the one that maps to ITL: a step emits the verified token
                plus whatever drafts survived, so 1.0 means spec decode bought
                nothing and k+1 is the ceiling.
    """
    drafts = counter(mm, "vllm:spec_decode_num_drafts")
    dtok = counter(mm, "vllm:spec_decode_num_draft_tokens")
    acc = counter(mm, "vllm:spec_decode_num_accepted_tokens")
    if dtok is None and acc is None:
        # NATIVE ATOM publishes the same two totals under atom:mtp_*. It has no
        # draft-COUNT metric, but a NEXTN/mtp step drafts exactly
        # num_speculative_tokens tokens every time, so drafts = dtok / k is
        # exact rather than an estimate. k is recovered from the server's own
        # cmdline below; without it tok/step would be unreportable.
        dtok = counter(mm, "atom:mtp_draft_tokens")
        acc = counter(mm, "atom:mtp_accepted_tokens")
        k = spec_stats.num_spec_tokens
        drafts = (dtok / k) if (dtok and k) else None
    if not drafts or not dtok or acc is None:
        return None
    per = {}
    for s in mm.get("vllm:spec_decode_num_accepted_tokens_per_pos", {}).get("series", []):
        pos = s.get("labels", {}).get("position")
        tot = s.get("stats", {}).get("total")
        if pos is not None and tot is not None:
            per[int(pos)] = per.get(int(pos), 0.0) + tot
    return {"drafts": drafts, "dtok": dtok, "acc": acc,
            "accept": 100.0 * acc / dtok, "tok_step": 1.0 + acc / drafts,
            # per position: share of drafts whose token at that position was
            # accepted. Denominator is drafts, not dtok -- every draft offers
            # exactly one token per position.
            "per_pos": {p: 100.0 * v / drafts for p, v in sorted(per.items())}}


# Speculation depth k, for the atom:mtp_* branch above, which has no draft-count
# metric to divide by. main() sets it from the server's recorded cmdline. Stays
# None when the run has no --num-speculative-tokens, which is also when the
# branch cannot be reached: no spec config, no atom:mtp_* metrics.
spec_stats.num_spec_tokens = None


def point(cdir, gpus):
    """One concurrency point -> row dict, or None if it has no usable records."""
    jl = [p for p in glob.glob(os.path.join(cdir, "**", "*.jsonl"), recursive=True)
          if "prompts" not in os.path.basename(p)]
    if not jl:
        return None
    # WARMUP RECORDS ARE NOT MEASUREMENTS AND MUST NOT BE SCORED.
    # The harness passes --warmup-request-count 3, and aiperf tags each record
    # with metadata.benchmark_phase ("warmup" or "profiling") and excludes the
    # warmup ones from its own reported percentiles. This function used to
    # filter on input_sequence_length alone, which kept them -- so every report
    # scored 3 requests the instrument had already disowned, and the run's row
    # disagreed with profile_c<conc>.json for no visible reason.
    #
    # It mattered most exactly where the targets are tightest. Warmup requests
    # prefill into an empty prefix cache, so they are the slowest in the run
    # (conc 8, 2026-09-25: warmup TTFT p90 23,491 ms against 2,282 ms for the
    # 80 profiling requests). At a low point n is small enough that 3 outliers
    # fall inside the top decile and land on the p90 boundary itself: conc 8
    # read 9,734 ms with them and 2,282 ms without -- the difference between
    # missing the 5 s TTFT target and meeting it comfortably. Higher points
    # moved under 6 %, which is why this survived unnoticed.
    #
    # Excluding them also fixes the throughput window: t0/t1 below span the
    # records kept, so warmup previously stretched the duration while its
    # tokens were counted too. Both now come from the profiling phase alone,
    # which is the window aiperf reports against.
    #
    # Records predating this metadata have no benchmark_phase; those are KEPT,
    # so an older results tree still renders instead of silently emptying.
    recs = []
    for line in open(jl[0]):
        if not line.strip():
            continue
        r = json.loads(line)
        if "input_sequence_length" not in r.get("metrics", {}):
            continue
        if r.get("metadata", {}).get("benchmark_phase") == "warmup":
            continue
        recs.append(r)
    if not recs:
        return None

    M = lambda r, k: r["metrics"][k]["value"]
    t0 = min(r["metadata"]["request_start_ns"] for r in recs)
    t1 = max(r["metadata"]["request_end_ns"] for r in recs)
    dur = (t1 - t0) / NS
    isl = [M(r, "input_sequence_length") for r in recs]
    osl = [M(r, "output_sequence_length") for r in recs]
    ttft = [M(r, "time_to_first_token") for r in recs]
    itl = [M(r, "inter_token_latency") for r in recs if "inter_token_latency" in r["metrics"]]

    cache, spec = None, None
    sm = glob.glob(os.path.join(cdir, "**", "*_server_metrics.json"), recursive=True)
    if sm:
        mm = json.load(open(sm[0])).get("metrics", {})
        q, h = counter(mm, "vllm:prefix_cache_queries"), counter(mm, "vllm:prefix_cache_hits")
        if q:
            cache = 100.0 * h / q
        else:
            # NATIVE ATOM (atom.entrypoints.openai_server) names its metrics
            # atom:*, not vllm:*, so the block above finds nothing and the
            # column used to render "-" for a server that does publish the
            # number. The pair is cached/full, NOT cached/wanted: wanted is
            # clamped to what was actually cached, so cached/wanted is
            # identically 1.0 and silently reports a 100 % hit rate.
            # full_tokens is the prompt tokens the requests asked for --
            # 9,200,959 against 80 x 115,000 on the run this was derived from.
            c, f = (counter(mm, "atom:prefix_cache_cached_tokens"),
                    counter(mm, "atom:prefix_cache_full_tokens"))
            if f:
                cache = 100.0 * c / f
        spec = spec_stats(mm)

    return {
        "conc": int(os.path.basename(cdir).rsplit("_", 1)[-1]),
        "reqs": len(recs), "dur": dur,
        "isl": sum(isl) / len(isl), "osl": sum(osl) / len(osl), "cache": cache,
        # p90 IS THE SCORED LATENCY (requirements.txt, 2026-09-25: P90 TTFT < 5 s,
        # P90 ITL <= 20 ms), so it leads each pair in the table. p50 is kept
        # beside it because the gap between them is the diagnosis -- a point
        # whose p50 passes and p90 does not is one where a minority of requests
        # decode alongside prefills.
        #
        # ITL p90 is over per-REQUEST mean inter-token latency, the same
        # population as p50 -- aiperf reports one inter_token_latency per
        # request, not one per token. So it is the 90th-percentile *request*,
        # not the 90th-percentile token gap, and it does not bound a single
        # stall.
        "ttft50": pct(ttft, 50), "ttft90": pct(ttft, 90),
        "itl50": pct(itl, 50), "itl90": pct(itl, 90),
        "in_gpu": sum(isl) / dur / gpus, "out_gpu": sum(osl) / dur / gpus,
        "spec": spec,
    }


def resolve(script):
    """Extract env exports + the main command from a wrapper script, with
    shell variables substituted so the result stands alone."""
    if not script or not os.path.exists(script):
        return None
    text = open(script).read()
    var = {}
    for m in re.finditer(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.+)$", text, re.M):
        v = m.group(2).split("#")[0].strip().strip('"').strip("'")
        d = re.match(r"^\$\{[A-Z_][A-Z0-9_]*:-(.*)\}$", v)
        var[m.group(1)] = d.group(1) if d else v

    def sub(s):
        # quoted forms first, so "$PORT" collapses to 8330 rather than "8330"
        s = re.sub(r'"\$\{([A-Z_][A-Z0-9_]*):-(.*?)\}"', lambda m: var.get(m.group(1), m.group(2)), s)
        s = re.sub(r'"\$\{?([A-Z_][A-Z0-9_]*)\}?"', lambda m: var.get(m.group(1), m.group(0)), s)
        s = re.sub(r"\$\{([A-Z_][A-Z0-9_]*):-(.*?)\}", lambda m: var.get(m.group(1), m.group(2)), s)
        s = re.sub(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", lambda m: var.get(m.group(1), m.group(0)), s)
        return s.replace(' "$@"', "")

    out = [sub(l.rstrip()) for l in text.splitlines()
           if re.match(r"^export\s+[A-Z]", l) and "PATH" not in l]
    lines, grab = [], False
    for l in text.splitlines():
        # `exec ` is optional and must be tolerated: both serve scripts exec the
        # server so it replaces the shell and keeps the PID the client records
        # into arm.json. Without it here, resolve() returned no command at all,
        # which silently emptied the Server block AND broke the entrypoint
        # tie-break in pick() -- it matches against this same resolved text.
        if re.match(r"^\s*(?:exec\s+)?(vllm serve"
                    r"|python3? -m atom\.entrypoints\.openai_server"
                    r"|python3? -m sglang\.launch_server"
                    r"|\./run_gsm8k_benchmark\.sh|aiperf )", l):
            grab = True
        if grab:
            # Drop the `exec ` prefix from the rendered block: it is how the
            # script hands its PID to the server, not part of the command a
            # reader reproduces the run with.
            lines.append(re.sub(r"^(\s*)exec\s+", r"\1", sub(l.rstrip()), count=1))
            if not l.rstrip().endswith("\\"):
                break
    return "\n".join(out + ([""] if out and lines else []) + lines).strip() or None


def server_argv(results_dir):
    """The server's raw argv from arm.json, or [] if the run predates it."""
    try:
        with open(os.path.join(results_dir, "arm.json")) as fh:
            return json.load(fh).get("server_cmdline") or []
    except (OSError, ValueError):
        return []


def recorded_server_args(results_dir):
    """The server's ACTUAL --flag/value pairs, from arm.json's server_cmdline.

    resolve() above reconstructs the server block by substituting a wrapper
    script's `${VAR:-default}` literals. That is right when nothing overrode
    them and silently wrong when something did: a run driven by
    `GPU_MEM_UTIL=0.86 ./serve_...sh` renders as 0.90, because 0.90 is what the
    file says. On 2026-09-27 that put a copy-pasteable command at the melting
    utilisation inside the very report that demonstrates 0.86 is safe.

    Only arm.json written by a client new enough to record server_cmdline has
    this; older runs return {} and keep the previous behaviour.
    """
    argv = server_argv(results_dir)
    args, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--"):
            if "=" in argv[i]:
                k, v = argv[i].split("=", 1)
                args[k] = v
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                args[argv[i]] = argv[i + 1]
                i += 1
            else:
                args[argv[i]] = None          # bare switch
        i += 1
    return args


def repo_line(results_dir):
    """The measured software, from arm.json: aiter / ATOM / vLLM shas.

    A number is a fact about one server config AND one build. aiter and ATOM
    are `pip install -e`'d, so they are read live at the next server start --
    an edit sitting in either worktree is already in the measurement with
    nothing in the report to say so. arm.json records the sha and a `dirty`
    flag per repo, written by the client BEFORE the run; this renders them.

    `dirty` is the load-bearing half: against an uncommitted worktree the sha
    alone is a lie, so it is marked rather than dropped. Returns None for a
    results dir with no arm.json (older runs) so the report is unchanged
    rather than carrying an empty line.
    """
    try:
        with open(os.path.join(results_dir, "arm.json")) as fh:
            arm = json.load(fh)
    except (OSError, ValueError):
        return None
    parts = []
    for label, key in (("aiter", "aiter"), ("ATOM", "ATOM"), ("vLLM", "vllm")):
        r = arm.get(key)
        if not isinstance(r, dict) or not r.get("sha"):
            continue
        sha = r["sha"]
        sha = sha[:9] if sha != "unknown" else sha
        parts.append(f"{label} `{sha}`" + (" **dirty**" if r.get("dirty") else ""))
    if not parts:
        return None
    line = " | ".join(parts)
    if arm.get("vllm_version"):
        line += f" | vllm {arm['vllm_version']}"
    return line


def reconcile(block, recorded):
    """Correct rendered flag values against what the server actually ran.

    Returns (block, [notes]). The comparison is quote-aware: the script writes
    --compilation-config '{"cudagraph_mode": ...}' and /proc hands back the same
    JSON unquoted, so a naive token compare calls every quoted argument a
    mismatch. Values that carry their own quoting are compared but never
    rewritten -- re-quoting a JSON blob textually is how a report would start
    lying in a *new* way; a genuine difference there is reported as a note.
    """
    if not block or not recorded:
        return block, []
    fixed = []
    for flag, val in recorded.items():
        if val is None:
            continue
        m = re.search(re.escape(flag) + r"""[= ]('[^']*'|"[^"]*"|[^\s\\]+)""", block)
        if not m:
            continue
        shown = m.group(1)
        if shown[:1] in "'\"" and shown[-1:] == shown[:1]:
            shown = shown[1:-1]
        if shown == val:
            continue
        if re.search(r"""[\s'"]""", val) or shown != m.group(1):
            fixed.append("`%s` actually ran as `%s` (left as rendered: not a "
                         "simple value)" % (flag, val))
            continue
        block = block[:m.start(1)] + val + block[m.end(1):]
        fixed.append("`%s` corrected %s -> **%s**" % (flag, m.group(1), val))
    return block, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("-o", "--out")
    ap.add_argument("--gpus", type=int)
    ap.add_argument("--title")
    ap.add_argument("--server-script")
    ap.add_argument("--client-script")
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args()

    root = os.path.abspath(a.results)
    meta = {}
    if os.path.exists(os.path.join(root, "run_meta.json")):
        meta = json.load(open(os.path.join(root, "run_meta.json")))
    gpus = a.gpus or int(meta.get("gpus") or 8)

    # Must precede point(), which reads it via spec_stats(). Native ATOM reports
    # drafted/accepted token totals but no draft COUNT, so tok/step is only
    # recoverable from the depth the server was actually launched with -- and
    # arm.json is the only record of that. int() deliberately unguarded: a
    # non-numeric --num-speculative-tokens means arm.json disagrees with a server
    # that started, which is a bug to see, not to paper over with a dropped column.
    rec = recorded_server_args(root)
    if rec.get("--num-speculative-tokens"):
        spec_stats.num_spec_tokens = int(rec["--num-speculative-tokens"])

    rows = [r for r in (point(d, gpus) for d in sorted(glob.glob(os.path.join(root, "concurrency_*"))))
            if r]
    if not rows:
        sys.exit(f"no usable records under {root}")
    rows.sort(key=lambda r: r["conc"])

    proj = os.path.dirname(os.path.dirname(root))

    def pick(given, pat, flag):
        """Find the wrapper script that produced this run -- or refuse to guess.

        The tree holds several serve_*.sh (GLM-5.3 canonical, GLM-5.3 ptrace,
        GLM-5.2), so the old "first alphabetical match" would have rendered a
        GLM-5.2 server block under a GLM-5.3 run with nothing in the report
        saying so. A report's Server block is the only record of what produced
        the numbers, so an ambiguous match is an error, not a coin flip.

        Order: explicit flag, then the run's own serve_snapshot.sh (what
        snapshot_server.sh wrote next to the results, the only real witness),
        then a unique glob, then the candidates whose RESOLVED command names the
        model run_meta.json records -- comments are excluded on purpose, since
        every script mentions the others.
        """
        if given:
            return given
        snap = os.path.join(root, "serve_snapshot.sh")
        if pat.startswith("serve_") and os.path.exists(snap):
            return snap
        cands = sorted(glob.glob(os.path.join(proj, pat)))
        if len(cands) <= 1:
            return cands[0] if cands else None
        model = os.path.basename((meta.get("model") or "").rstrip("/"))
        named = [c for c in cands if model and model in (resolve(c) or "")]
        if len(named) == 1:
            return named[0]
        # WHICH ENTRYPOINT ran, from arm.json's server_cmdline. The two recipes
        # for this checkpoint -- `vllm serve` (plugin backend) and
        # `python -m atom.entrypoints.openai_server` (native) -- name the same
        # model, so the model test above cannot separate them and every report
        # would need --server-script by hand. /proc/<pid>/cmdline can: it says
        # which binary the measured server actually was.
        if pat.startswith("serve_"):
            argv = " ".join(server_argv(root))
            want = ("atom.entrypoints.openai_server" if
                    "atom.entrypoints.openai_server" in argv else
                    "vllm serve" if "serve" in argv else None)
            if want:
                byep = [c for c in (named or cands) if want in (resolve(c) or "")]
                if len(byep) == 1:
                    return byep[0]
        # Canonical arm vs the same arm with the torch profiler wired in: the
        # results dir says which one ran. prof_run.sh drops window.txt /
        # capture.txt / trace_files.txt beside a profiled capture and a plain
        # sweep has none of them, so this is evidence, not a preference.
        profiled = any(os.path.exists(os.path.join(root, f))
                       for f in ("window.txt", "capture.txt", "trace_files.txt"))
        tier = [c for c in named if ("profiler" in (resolve(c) or "").lower()) == profiled]
        if len(tier) == 1:
            return tier[0]
        sys.exit("report_md.py: cannot tell which %s produced %s.\n  %s\n"
                 "Pass %s explicitly (or snapshot the arm with snapshot_server.sh)."
                 % (pat, root, "\n  ".join(named or cands), flag))

    srv = resolve(pick(a.server_script, "serve_*.sh", "--server-script"))
    cli = resolve(pick(a.client_script, "client_*.sh", "--client-script"))

    # ...then correct it against what the server process actually ran. The
    # script only ever knew its own defaults; arm.json's server_cmdline is
    # /proc/<pid>/cmdline, which cannot disagree with the run.
    srv, srv_fixed = reconcile(srv, recorded_server_args(root))

    # The wrapper carries defaults; run_meta.json carries what this run actually
    # used (CLI overrides, $@). Prefer the latter so the block reproduces THIS run.
    #
    # --tokenizer is corrected from meta["model"] because run_meta.json has no
    # tokenizer field of its own. That is sound only while the client passes
    # --tokenizer "$MODEL"; the client says so in a comment for the same reason.
    # Without this, overriding MODEL renders a client block naming the previous
    # checkpoint's tokenizer -- a reproduction command that silently tokenizes
    # with the wrong vocabulary.
    if cli and meta:
        for flag, key in (("--concurrency", "concurrency"), ("--isl", "isl"), ("--osl", "osl"),
                          ("--cache", "cache"), ("--gpus", "gpus"), ("--url", "url"),
                          ("--model", "model"), ("--tokenizer", "model")):
            if meta.get(key) is not None:
                cli = re.sub(rf'({re.escape(flag)}\s+)(?:"[^"]*"|\S+)', rf'\g<1>{meta[key]}', cli)

    f = lambda v, s: format(v, s) if v is not None else "n/a"

    # WHICH NODE, in the header, derived -- never typed into a --note.
    # Two nodes exist and the same config has measured ~29 % apart on them
    # (CLAUDE.md), so a table without a node is unattributable. The series
    # directory is results_<node>_<series> and the node half is derived by the
    # client from visible HBM at run time, so it is already a fact about the
    # run rather than something a caller chose -- taking it from the path keeps
    # it that way. Anything hand-passed could be wrong; this cannot.
    node = None
    series = os.path.basename(os.path.dirname(root))
    if series.startswith("results_") and "_" in series[len("results_"):]:
        node = series[len("results_"):].split("_", 1)[0]

    L = [f"# {a.title or os.path.basename(root)}", ""]
    if meta:
        L += [f"`{meta.get('model')}` | {node + ' | ' if node else ''}{gpus} GPUs "
              f"| ISL {meta.get('isl')} / OSL {meta.get('osl')} "
              f"| `--cache {meta.get('cache')}` | seed {meta.get('seed')}", ""]
    # The build, beside the workload: same reason the node is in the header.
    # Two shas are two configurations even at identical flags.
    repos = repo_line(root)
    if repos:
        L += [repos, ""]
    # Spec-decode columns only when the server actually ran it; on a plain
    # server the metrics do not exist and empty columns would imply zero.
    has_spec = any(r["spec"] for r in rows)
    # Units live in the header, not in a legend line under the table -- the
    # report carries data and the commands that produced it, nothing else.
    hdr = ("| conc | reqs | dur(s) | ISL | OSL | cache% | TTFT p90 ms | TTFT p50 ms "
           "| ITL p90 ms | ITL p50 ms | in/s/gpu | out/s/gpu |")
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    if has_spec:
        hdr += " accept% | tok/step |"
        sep += "---:|---:|"
    L += [hdr, sep]
    for r in rows:
        line = (f"| {r['conc']} | {r['reqs']} | {r['dur']:.0f} | {r['isl']:.0f} | {r['osl']:.0f} "
                f"| {f(r['cache'],'.2f')} | {r['ttft90']:.0f} | {r['ttft50']:.0f} "
                f"| {f(r['itl90'],'.2f')} | {f(r['itl50'],'.2f')} "
                f"| {r['in_gpu']:.1f} | {r['out_gpu']:.2f} |")
        if has_spec:
            s = r["spec"]
            line += (f" {f(s and s['accept'], '.2f')} | {f(s and s['tok_step'], '.3f')} |")
        L.append(line)
    # Gated on npos, not has_spec: the per-position breakdown comes from
    # vllm:spec_decode_num_accepted_tokens_per_pos, which native ATOM does not
    # publish. Keyed off has_spec alone, an ATOM run emitted a header with
    # zero position columns, a separator one column short of it, and a lone
    # drafts figure the accept%/tok/step columns above already imply.
    npos = max((len(r["spec"]["per_pos"]) for r in rows if r["spec"]), default=0)
    if has_spec and npos:
        L += ["", "## Accepted drafts by position (% of drafts)", ""]
        L += ["| conc | " + " | ".join(f"pos {i}" for i in range(npos)) + " | drafts |",
              "|---:|" + "---:|" * (npos + 1)]
        for r in rows:
            s = r["spec"]
            if not s:
                continue
            cells = " | ".join(f(s["per_pos"].get(i), ".1f") for i in range(npos))
            L.append(f"| {r['conc']} | {cells} | {s['drafts']:.0f} |")
    for n in a.note:
        L += ["", n]
    if srv:
        L += ["", "## Server", "", "```bash", srv, "```"]
        if srv_fixed:
            # Say that the block was corrected, and to what. A silent rewrite
            # would be a second way for this block to be unverifiable.
            L += ["", "Corrected against the server's own `/proc/<pid>/cmdline` "
                  "(the script's defaults differed from what ran): "
                  + "; ".join(srv_fixed) + "."]
    if cli:
        L += ["", "## Client", "", "```bash", cli, "```"]

    md = "\n".join(L) + "\n"
    out = a.out or os.path.join(root, "report.md")
    open(out, "w").write(md)
    print(md)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

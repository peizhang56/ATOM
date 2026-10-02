# CLAUDE.md — DSpark on ATOM's vLLM plugin for DeepSeek-V4-Pro-0813

Re-design and re-implementation, starting from a clean tree. 2026-10-02.

The previous attempt is described in `SESSION-HANDOFF.md`. It is kept for its
*measurements*, which are sound, and for its list of dead ends. Its
**architecture is not the one to rebuild** — see §2.

---

## 0. Durability — push everything, the node is disposable

**This node can be lost at any time.** `/home/pzhang12/deepseek2` is NOT a git
repository; `/app/ATOM` and `/app/vllm` are checkouts that only persist as long
as the node does. Nothing here survives on its own.

The fork is the backup: `upstream` = `git@github.com:peizhang56/ATOM.git`
(authenticated by SSH key *and* `gh` as `peizhang56`). `origin` is
`ROCm/ATOM.git` — **never push there**; it is the real upstream project.

Two branches hold this work. Keep both current:

| branch on `upstream` | holds |
|---|---|
| `ds_v4_atom_vllm_dspark_draft_t7fix` | the **code**: ATOM-owned DSpark draft + the block-width fix |
| `ds_v4_dspark_worklog` | the **worklog**: this file, `SESSION-HANDOFF.md`, serve/stop scripts, `accept_probe.py`, server logs. An orphan branch — no ATOM code history. |

Both branches live in the **same** fork repo, so one clone restores everything.
The worklog branch has unrelated history (its own `git init`); that is fine for
a worktree.

```bash
git clone git@github.com:peizhang56/ATOM.git /app/ATOM && cd /app/ATOM
git remote add upstream git@github.com:peizhang56/ATOM.git   # same URL as origin here
git remote set-url --push origin DISABLED                    # don't push to a fork by accident
git checkout ds_v4_atom_vllm_dspark_draft_t7fix && pip install -e .
git worktree add /home/pzhang12/deepseek2 ds_v4_dspark_worklog
```

(On *this* node `origin` is `ROCm/ATOM.git` and `upstream` is the fork, which
is backwards from the usual convention — check `git remote -v` before pushing
anywhere, every time.)

**Push at the end of every working session, and before any reboot or
long-running job.** A finding that exists only in this node's filesystem is a
finding you are about to lose. The worklog branch is cheap — logs included, the
whole thing is ~210 KB.

```bash
# code
cd /app/ATOM && git push upstream HEAD
# worklog  (remote name is `upstream` on this node, `origin` on a fresh clone)
cd /home/pzhang12/deepseek2 && git add -A && git commit -m "worklog: <what>" \
  && git push upstream HEAD:ds_v4_dspark_worklog
```

Keep the two in step: the worklog cites commit SHAs from the code branch, so
push the code first.

---

## 1. The directive

Stated by the user, 2026-10-02:

1. Read `SESSION-HANDOFF.md` for background.
2. Days were spent on the previous approach and it did not work.
3. `/app/ATOM` has been **reverted to clean `main`** — every commit from the
   previous effort is gone from the working tree. Re-design and re-implement
   ATOM support for `/data/DeepSeek-V4-Pro-0813` under **vllm_atom (plugin
   mode) with DSpark**.
4. Use `serve_dsv4_pro_atom_native_tp8.sh` to understand the **native** ATOM
   code path. Add logging if that is what it takes.
5. Understand how `/app/vllm` works with the ATOM plugin.
6. **The existing implementation on
   `github.com/peizhang56/ATOM/commits/ds_v4_atom_vllm/` is not correct: it
   runs inference in the ATOM plugin but runs DSpark from the vLLM side.**
7. This file documents the above. The goal is a *correct* implementation that
   enables the **vllm_atom plugin with DSpark** — ATOM owning the draft, not
   just the target.

Point 6 is the whole task. Everything below exists to make it precise.

---

## 2. Ground truth — verified against the trees on 2026-10-02

### 2.1 Repo state

| | |
|---|---|
| `/app/ATOM` | branch `main` @ `922b35196`, clean. **`pip install -e`** — edits and the checked-out branch are live for every process on the node. |
| `/app/vllm` | `281cfd5a09`, clean. **NOT editable** — the runtime copy is `/opt/venv/lib/python3.12/site-packages/vllm` (`0.28.1.dev0+g2cf0a6915`). Editing `/app/vllm` changes nothing at runtime. The files cited in this doc were diffed and are byte-identical between the two. |
| previous work | `upstream/ds_v4_atom_vllm` (half-finished), `upstream/ds_v4_atom_vllm_dspark_draft` (abandoned), `upstream/ds_v4_atom_vllm_dspark_draft_t7fix` (**the live branch** — see §6). `be473f7a3` is not an ancestor of `main`. |

So DS-V4 + DSpark in plugin mode does not exist **on `main`**. It does exist,
and substantially works, on `_t7fix` — §6 explains why the plan is to finish
that stack rather than restart on `main`.

What *does* exist on clean `main`:

- `atom/plugin/vllm/deepseek_v4_bridge.py` (2182 lines) — the V4 proxy
  attention layer, the proxy KV arena, and the ATOM↔vLLM metadata translation.
  DS-V4 **target** inference in plugin mode works today.
- `atom/plugin/vllm/models/deepseek_v4.py` — the plugin's V4 target subclass
  (vLLM attention/indexer variants, mixed-batch indexer split).
- `atom/plugin/vllm/models/deepseek_v4_mtp.py` — the plugin's V4 **MTP** draft.
  Precedent: an ATOM draft already runs under vLLM for this model.
- `atom/models/deepseek_v4_dspark.py` (1408 lines) — ATOM's **own** DSpark
  draft for V4. Mature, used by native, **unreachable from plugin mode**.
- `atom/spec_decode/dspark_proposer.py` (1068 lines) — ATOM's native DSpark
  drafting loop.

What does **not** exist on clean `main`:

- Any aux-hidden-state emission in `atom/models/deepseek_v4.py` (`grep aux` →
  nothing). The target cannot feed a DSpark draft yet.
- Any registration of a V4 DSpark draft into vLLM's model registry.

### 2.2 The two arms, and where the seam actually falls

**Native** (`serve_dsv4_pro_atom_native_tp8.sh`): ATOM owns everything —
scheduler, KV manager, target, drafter, draft model.

**Plugin as previously shipped** (`ds_v4_atom_vllm`, `be473f7a3`):

| component | owner |
|---|---|
| scheduler, KV cache manager | vLLM |
| target model (61 layers) + attention | **ATOM** (V4 proxy bridge) |
| DSpark speculator (block layout, Markov loop, rejection) | vLLM |
| **DSpark draft model** | **vLLM** — `vllm/models/deepseek_v4/amd/dspark.py` |

That last row is the defect. `atom/models/deepseek_v4_dspark.py` — ATOM's own,
tuned, measured draft — was never wired in. vLLM's draft was used instead.

### 2.3 What "correct" means here

vLLM's `DSparkSpeculator` **is** the plugin contract, not a thing to displace.
ATOM already ships this exact shape for Kimi-K3:

- `atom/plugin/vllm/models/kimi_k3_dspark.py` → `KimiK3DSparkVllm`
- registered in `register.py` as `"K3DSparkModel"` →
  `atom.plugin.vllm.models.kimi_k3_dspark:KimiK3DSparkVllm`
- vLLM's speculator drives the loop; **ATOM supplies the draft model**.

So "ATOM owns DSpark" means **ATOM owns the draft model and its KV**, driven by
vLLM's speculator — the same division that ships for K3 and for V4's MTP draft.
It does **not** mean porting `DSparkProposer` into vLLM's V2 model runner.
(`atom/spec_decode/dspark_proposer.py` is driven by ~30 call sites in ATOM's
own 4700-line `model_runner.py`; replacing vLLM's speculator would mean
reimplementing all of it against vLLM's `InputBatch`/`BlockTables`. Not the
job, and not what the K3 precedent does.)

### 2.4 The exact contract to implement

vLLM's V2 GPU model runner is forced on for `method == "dspark"`
(`vllm/config/vllm.py:628`), so the live speculator is
`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (`DSparkSpeculator` ←
`DFlashSpeculator`), selected in `vllm/v1/worker/gpu/spec_decode/__init__.py`.
It calls exactly these on the draft model, plus `__call__`:

```
combine_hidden_states        precompute_and_store_context_kv
compute_draft_logits         compute_confidence
markov_embed                 markov_bias
apply_markov_bias_gathered   map_draft_to_target
get_draft_kv_cache_layer_names   get_draft_attn_causal
```

The draft architecture vLLM resolves is **`"DSparkDraftModel"`** →
`vllm.models.deepseek_v4:DSparkDeepseekV4ForCausalLM`
(`vllm/model_executor/models/registry.py:637`). Overriding that key in
`_VLLM_MODEL_REGISTRY_OVERRIDES` is how ATOM takes it — the same one-line move
that already claims `"K3DSparkModel"`.

### 2.5 Checkpoint facts (`/data/DeepSeek-V4-Pro-0813/config.json`)

```
dspark_block_size      = 5          # trained block γ
dspark_target_layer_ids= [58,59,60] # aux taps the draft's main_proj consumes
dspark_markov_rank     = 512
dspark_noise_token_id  = 128799
sliding_window         = 128        # the draft's rolling target-KV window
index_topk             = 1024 ; hc_mult = 4 ; num_hidden_layers = 61
sample_from_anchor     — ABSENT
```

Draft weights live in the target checkpoint under `mtp.{0,1,2}.*` — three full
DSpark backbone layers (attn + MoE + mHC), with `mtp.0.main_proj/main_norm`,
`mtp.2.markov_head.markov_w{1,2}`, `mtp.2.confidence_head.proj`, and
`mtp.2.hc_head_*`/`mtp.2.norm`. Embedding and LM head are shared with the
target.

Both arms run `num_speculative_tokens=7`, which exceeds `dspark_block_size=5`.
ATOM warns about this and accepts it (`atom/config.py:2437`).

---

## 3. The block width — settled by measurement

T = `num_speculative_tokens` = **7**. This is no longer an inference; it was
read off vLLM's own speculator object at runtime (2026-10-02, TP8, this
checkpoint):

```
ATOM DIAG: speculator=DSparkSpeculator num_query_per_req=7
           num_speculative_steps=7 sample_from_anchor=True
           anchor_idx[:4]=[0, 7, 14, 21]
```

The abandoned branch used `1 + num_speculative_tokens` = 8, so it strided 8
over a width-7 layout. **But correcting it to 7 is necessary and not
sufficient** — see §3.1. Measured, same probe (concurrency 32, ~80-token
prompts, 256 output tokens) on both:

| arm | cudagraphs | result |
|---|---|---|
| T=8 (branch as written) | FULL_DECODE_ONLY | serves, 32/32, 8192 tokens, 954 tok/s, zero faults — but drafts from wrong anchors |
| T=7 (corrected) | FULL_DECODE_ONLY | **`Memory access fault` on all 8 GPUs**, engine dead |
| T=7 (corrected) | `--enforce-eager` | **works**: zero faults, **mean acceptance 2.50** at conc 32 |

**The width fix is real.** With T=7 in eager mode the ATOM-owned draft runs
correctly and acceptance is 2.50 (2421 accepted / 11263 drafted), per-position
`0.597, 0.382, 0.228, 0.144, 0.091, 0.044, 0.019` — a healthy decaying block,
against the ~1.10 the branch reported at T=8. In the same eager run
`num_blocks` tracked the live request count exactly (4, 3, 3, 2, 1 as requests
drained), confirming `total // 7` recovers the batch.

(Probe regime: ~80-token prompts, concurrency 32, 192 output tokens. **Not**
the ISL-115k regime the handoff's 3.3/2.38 numbers come from, so compare the
shape of the result, not the absolute number, against those.)

### 3.1 The remaining defect is cudagraph-specific

T=7 faults under `cudagraph_mode=FULL_DECODE_ONLY` and is clean under
`--enforce-eager`, same code and same probe. So it is **not** the block width,
**not** arena sizing, and **not** the draft's KV binding — all three were
instrumented and came back consistent:

```
ATOM DIAG: bind ready=True n_stages=3 stage_layer_ids=[61, 62, 63]
           n_compress_ratios=64 draft_args_n_layers=61
           unified_kv_bound=[True, True, True]
           window=WindowParams(ring_start=34380, slot_rows=43746,
                               ring_slots=135, ring_stride=135, run_rows=405)
```

Note `len(compress_ratios) == 64` for `num_hidden_layers == 61`: the
checkpoint's `compress_ratios` already carries three trailing `0` (dense)
entries for the draft stages, so the draft's stages 61/62/63 index the arena's
per-layer view list validly and `ring_slots == 135 == sliding_window + 7`. An
earlier theory that the arena was sized for 61 layers while the draft bound 64
is **wrong** — both sizing and carve read the same 64-entry list.

**Root cause — confirmed by reading vLLM, consistent with all three arms.**
`DFlashSpeculator.propose` pads the batch to a cudagraph bucket and hands the
draft model the *padded* token count:

```python
num_reqs_padded  = batch_desc.num_reqs or num_reqs
num_tokens_padded = batch_desc.num_tokens      # = num_reqs_padded * num_query_per_req
...
self._generate_draft(num_reqs, num_tokens_padded, ...)   # -> self.model(...[:num_tokens_padded])
```

The ATOM wrapper derives its batch as `num_blocks = total // T`, which is
therefore **`num_reqs_padded`, not the live request count**. vLLM's own draft
models tolerate this because they are token-parallel — every row is
independent. ATOM's DSpark draft is *request*-parallel: it rebuilds each block
from an anchor and indexes per-request state slots. Blocks
`num_reqs .. num_reqs_padded-1` address slots that were never allocated →
out-of-bounds read → `Memory access fault`.

This explains all three arms exactly:

- **eager**: no capture, so `batch_desc` returns the unpadded count,
  `num_blocks == num_reqs`. Clean — and the measured block counts tracked live
  requests (4, 3, 3, 2, 1) precisely as this predicts.
- **T=7 + cudagraphs**: `num_blocks == num_reqs_padded > num_reqs`. Faults.
- **T=8 + cudagraphs**: `num_blocks = num_reqs_padded * 7 // 8 < num_reqs_padded`,
  so it under-runs the batch and stays in bounds. It never faults *because* it
  is wrong — which is why the width bug hid this one.

**The fix, and the trap in it.** ATOM already owns the right mechanism:
`DSparkIndexBuffers.mask_pad_tail(row_ids, real_batch, batch)` in
`atom/model_ops/v4_kernels/dspark_fp8_indices.py` marks padded rows not-real
via a device-side `bid >= 0` sentinel, and the bridge already reasons about "a
sentinel tail for padded draft slots" (`deepseek_v4_bridge.py:1546`). The draft
path must mark blocks `>= num_reqs` the same way.

The trap: **the draft's forward is inside the captured graph**, so the real
batch count cannot be a Python `int` read at trace time — it would freeze at
the capture value and the sentinel would be wrong on every replay. This is the
identical hazard `DeepseekV4ModelVllm._mtp_hidden_buffer` exists to dodge
(stable-address buffer refreshed by an *in-graph* `copy_`). The real batch
count must reach the draft as a **device tensor refreshed each step**, not an
attribute.

Secondary suspect if that does not fully explain it:
`_demote_piecewise_cudagraph` silently turning the requested
`FULL_AND_PIECEWISE` into `FULL_DECODE_ONLY` on this path (handoff §1).

**Do not read the old ~1.10 tok/step as "ATOM's drafter is slow".** It is the
number produced by a drafter reading its anchors from MASK rows.

Reproduce: `verify-t7-diagnostics.patch` in this directory applies to
`upstream/ds_v4_atom_vllm_dspark_draft` and contains the width fix plus the
(temporary, clearly-marked) diagnostics. Logs: `logs/server-{control-t8,
diag-t7,eager-t7}.log`.

---

## 3.2 Historical note — how the width error was originally reasoned

Kept because the reasoning is the thing to avoid repeating.

`sample_from_anchor` is **absent** from this checkpoint's config, and vLLM
defaults it to `True` (`dspark/speculator.py:49`). Therefore:

```python
self.num_query_per_req = self.num_speculative_steps   # 7, NOT 1 + 7
self._anchor_idx = arange(max_num_reqs) * self.num_query_per_req
```

The anchor is column 0 of each **width-7** block; the anchor position is itself
the first prediction (`vllm/config/vllm.py:606-611`).

Native ATOM agrees exactly: `DSparkProposer._resolve_mtp_k` returns
`num_speculative_tokens` (= 7), and `dspark_proposer.py:277` comments
`self.mtp_k,  # max_seqlen_qo = block width T`. **T = 7 on both arms.**

`ds_v4_atom_vllm_dspark_draft` used `return 1 + num_speculative_tokens` → 8.
With stride 8 over a width-7 layout, request 0's anchor is right and **every
other request reads its anchor from a MASK/noise row**, and the returned
`normed` is 8 rows per block where vLLM indexes 7. That is a sufficient
explanation for the branch's measured ~1.10 accepted tok/step (≈ "first
request drafts, nobody else does"). The handoff lists this as a *known
remaining defect, untested* — it is almost certainly **the** defect, not one of
several.

Status: **the width half of this was confirmed on hardware (§3). The
conclusion that it was "*the* defect" was wrong** — it is one of at least two.
The lesson stands: measure before building on a code-reading diagnosis.

---

## 4. Target architecture

Two independent halves. Land and verify them separately.

### A. Target side — aux hidden states out of ATOM's V4

`atom/models/deepseek_v4.py` must optionally return per-layer aux hidden states
for `dspark_target_layer_ids`, mean-reduced over the mHC dimension, matching
what the draft was trained against. `be473f7a3` did this correctly and is worth
reading as a reference (`git show be473f7a3 -- atom/models/deepseek_v4.py`),
with its own caveat respected:

> The aux branch sits **inside** the `@support_torch_compile` region, so
> `aux_hidden_state_layers` is baked into the graph. From `--level 2` up the
> custom dispatcher replays code object 0 without evaluating guards, so it must
> be set at **load time and never mutated after**.

Prefer ATOM's drafter-owned `AuxCaptureSpec` hook mechanism
(`atom/spec_decode/drafter.py`) over a model-side branch **if** it can be made
to work under vLLM's compile/capture regime — it is how native taps the target
and it keeps the target model agnostic. If it cannot, the model-side branch is
the fallback, with the load-time-only constraint documented in-file.

### B. Draft side — ATOM's DSpark draft under vLLM's speculator

New file `atom/plugin/vllm/models/deepseek_v4_dspark.py`, modelled on
`kimi_k3_dspark.py` and `deepseek_v4_mtp.py`:

- `DeepseekV4DSparkDraft(DeepseekV4DSpark)` — rebind
  `deepseek_v4_base.DeepseekV4Attention`/`Indexer` to the vLLM variants during
  `__init__` (the established pattern), add the plain-backbone `forward` vLLM
  calls, **at T = num_speculative_tokens**.
- `DeepseekV4DSparkVllm(ATOMMoEForCausalLM)` — the ten-method contract from
  §2.4, `has_own_embed_tokens = has_own_lm_head = False`,
  `draft_id_to_target_id = None`.
- Register `"DSparkDraftModel"` in `_VLLM_MODEL_REGISTRY_OVERRIDES`
  (`atom/plugin/vllm/register.py`).

**KV ownership — now partly answered by §3.1.** The abandoned branch's choice
("ATOM owns the draft's KV; vLLM's draft KV group goes vestigial") was
measured to *bind correctly*: the draft's three stages took arena planes
61/62/63 with `ring_slots=135`, because the checkpoint's `compress_ratios`
already carries three trailing dense entries. So option 1 below is not just
preferred, it is substantially already working. Keep it.

ATOM's V4 DSpark draft does
*not* use a paged pool: each stage writes a **private per-request rolling SWA
ring** (`attn.swa_plane`, addressed by `attn.swa_window`), sized
`sliding_window` + draft width, and reads it by absolute position
(`deepseek_v4_dspark.py:653,715`). K3's draft is paged; V4's is not.

Two options:

1. **Carve the draft's ring into the V4 proxy arena** as additional per-slot
   `EntryField`s in `_v4_state_layout` (`deepseek_v4_bridge.py:116`). The arena
   is already per-slot (`num_slots = max_num_seqs`, `ring_slots =
   win_with_spec`) and already budgets `ring_extra = num_speculative_tokens`.
   One KV group, no second block size, prefix cache unaffected. **Preferred.**
2. Re-register a second vLLM KV group, as `dspark_draft_kv_patch.py` did in
   `be473f7a3`. That patch existed to serve *vLLM's paged* draft; for ATOM's
   ring-based draft it would be vestigial. Avoid.

The abandoned branch chose "ATOM owns the KV, vLLM's draft KV group goes
vestigial" and entered the ATOM forward context itself inside
`precompute_and_store_context_kv` (vLLM calls it outside any
`ATOMMoEForCausalLM.forward`). That part of its reasoning is sound and is worth
reusing — read its module docstring:
`git show upstream/ds_v4_atom_vllm_dspark_draft:atom/plugin/vllm/models/deepseek_v4_dspark.py`.

### Not in scope for correctness

DP attention (1.735×) and the FP4 indexer (1.176×) are the *performance* gap
between the arms, decomposed exactly in `SESSION-HANDOFF.md` §2/§4. They are
real and they are separate. Do not entangle them with getting DSpark correct.

---

## 5. Operating rules

**Editable installs.** ATOM and aiter are `pip install -e`'d. An uncommitted
edit is already in the next measurement, and the checked-out branch is what
*every* process on this node uses. Never leave a branch checked out that you
would not want someone else's run to use.

**Always `./stop.sh` before relaunching.** It verifies VRAM actually released;
a half-dead mp executor silently mis-sizes the next run's KV profile.

**Never modify `@support_torch_compile` files** (ATOM's standing rule;
`/app/ATOM/CLAUDE.md`). `atom/models/deepseek_v4_dspark.py` carries a COMPILE
BOUNDARY block at the top naming exactly which functions are traced — read it
before touching that file. Instrument at call sites instead:
`ModelRunner.run_model()`, `DSparkProposer.propose()`, or the plugin wrapper's
methods, which are outside every compiled region.

**`rm -rf /root/.cache/atom/*` before restarting** after code changes — stale
compile cache causes silent failures.

**Plugin-only env vars** (in `serve_dsv4_pro_vllm_atom_tp8.sh`, not needed
natively): `VLLM_ROCM_USE_AITER=1` is what unblocks the MXFP4 MoE oracle;
without it the draft dies on `No module named 'triton_kernels.matmul_ogs'`, a
traceback that reads like a dtype bug and is not. Pair with
`--moe-backend aiter`, which raises instead of silently falling through.

**Method errors from the previous session, worth not repeating:**
- Run the single-stream determinism control *first* — greedy decoding is not
  reproducible at long context on either arm, so any "A differs from B" probe
  at 115k needs "A differs from A" ruled out before it means anything.
- Check the **server's** `Running:` count, not the client's concurrency.
- Where two paths differ by *kernel*, ablate; do not model from bytes.

**Hardware.** 8× gfx950 (MI355X) are visible on this node. See the
`this-cluster` skill if work needs to go through Slurm instead.

---

## 6. The plan

### What changed about the framing

The task was set up as "re-implement from scratch on clean `main`". After
reading the branches and measuring, **that is the wrong shape of work.** The
correct implementation substantially exists, as an 8-commit stack:

```
origin/main
 ├── be473f7a3   target side: aux hidden states, prefix SWA rollback, proxy arena
 ├── 5f00ac68a, 27f3e9234      diagnostics (flag-gated, no runtime effect)
 ├── 943ee681d .. f32df74be    ATOM-owned DSpark draft  (4 commits)
 └── b2fa562b9   block-width fix                        ← ds_v4_atom_vllm_dspark_draft_t7fix
```

The reviewer's "not correct" applies to **`ds_v4_atom_vllm`**, which stops at
vLLM's draft model. It is not *wrong*, it is **half-finished**: its target-side
work (ATOM's V4 emitting aux hidden states for layers 58/59/60) is exactly what
ATOM's own draft needs too. Throwing it away and starting over would rebuild
`be473f7a3` line for line. The `_t7fix` branch is the other half, and it is
measured working in eager at 2.50 acceptance.

So: **finish and validate this stack**, do not restart. The one thing standing
between it and a usable arm is the cudagraph fault, whose root cause is now
known (§3.1).

### Stage 1 — the cudagraph fault (blocker)

The only thing preventing a usable arm. Root cause and the in-graph trap are in
§3.1; the fix is to mark draft blocks `>= num_reqs` as not-real via ATOM's
existing `mask_pad_tail` sentinel, with the real batch count reaching the draft
as a **device tensor refreshed per step**, never a Python attribute.

- Reproduce first: `serve_dsv4_pro_vllm_atom_tp8.sh` + `accept_probe.py
  --concurrency 32`. Faults within ~40 s.
- Bisect cheaply before coding: run `--cudagraph-mode FULL_DECODE_ONLY` vs
  `PIECEWISE` vs `--enforce-eager`. Confirms the fault tracks *capture*, and
  tells you whether PIECEWISE alone is a usable interim arm.
- Instrument `num_reqs` vs `num_blocks` on real steps (the earlier diagnostic
  capped out during capture and never saw a real step — raise the cap and log
  only when `num_blocks != num_reqs`).

**Exit:** plugin arm serves at `FULL_AND_PIECEWISE`, concurrency 128, zero
faults, acceptance ≥ the eager number. Push.

### Stage 2 — delete the vestigial draft KV group

`dspark_draft_kv_patch.py` is still live (`spec_decode_patch.py:334`). It
registers a **second vLLM KV group** (sliding-window MLA, block 64) for
*vLLM's paged draft*. ATOM's draft does not use it — it writes its own SWA ring
in the proxy arena. So today the arm reserves a whole KV group for nothing.

This is not cleanup, it is capacity: those blocks come out of the same budget,
and `be473f7a3` had to convert the prefix-cache rollback *per group* because
the two differ in block size. Removing it should raise usable KV and simplify
the prefix path — both of which the handoff showed move throughput.

**Exit:** group gone, KV blocks measurably up, GSM8K unchanged, prefix hit rate
no worse. Measure KV capacity before and after.

### Stage 3 — correctness

Speculative decoding must be **lossless**. Verify in this order:

1. **Determinism control first** (handoff §3, the error that nearly became a
   root cause): same prompt, batch 1, twice, spec OFF. Establish the noise
   floor *at the context length you intend to test*. Greedy is not reproducible
   above ~15k on either arm, so do accuracy work at **short context**.
2. Greedy outputs, spec ON vs spec OFF, short context — should match
   token-for-token.
3. `lm_eval` GSM8K vs the plugin arm with speculation disabled, and vs native
   (handoff baseline 0.9583).

**Exit:** GSM8K within noise of the no-spec plugin arm; (2) matches exactly.

### Stage 4 — performance, at the regime the claim was made in

Everything measured so far is ~80-token prompts. The original premise — that
vLLM's drafter loses acceptance with batch (3.31 @ conc 16 → 2.38 @ 128) while
ATOM's holds ~3.3 — is an **ISL 115k** claim and is still unverified for ATOM's
drafter under the plugin.

Sweep concurrency 16 / 32 / 64 / 128 at ISL 115k / OSL 1k, three arms, same
client and seed: native, plugin + vLLM draft (`ds_v4_atom_vllm`), plugin + ATOM
draft (`_t7fix`). Report acceptance and in/s/gpu.

**Exit:** a table that either confirms or kills the premise. If ATOM's drafter
also decays with batch, the premise was wrong and §4.3 of the handoff is still
open — say so rather than tuning around it.

### Stage 5 — productionize

Follow `/app/ATOM/.claude/commands/add-atom-vllm-model.md`: recipe under
`recipes/atom_vllm/`, accuracy table with the raw JSON path, nightly CI entry
(not the PR matrix), `tests/plugin` no worse than main's baseline. Then rebase
the stack onto current `main` and open the PR.

### Explicitly not in this plan

DP attention (1.735×) and the FP4 indexer (1.176×) — the decomposed performance
gap (handoff §2/§4). Both are real, both are large, and both are **separate
from making DSpark correct**. Do not entangle them. `2ff2fd961` (FP4
groundwork, gated off) is deliberately not in the `_t7fix` stack.

---

## 7. Files that matter

| path | what |
|---|---|
| `atom/plugin/vllm/register.py` | `_VLLM_MODEL_REGISTRY_OVERRIDES` — where `"DSparkDraftModel"` goes |
| `atom/plugin/vllm/models/kimi_k3_dspark.py` | **the template** for the draft wrapper |
| `atom/plugin/vllm/models/deepseek_v4.py` | V4 target under vLLM (attention + indexer variants) |
| `atom/plugin/vllm/models/deepseek_v4_mtp.py` | the minimal "ATOM draft under vLLM" wrapper |
| `atom/plugin/vllm/deepseek_v4_bridge.py` | proxy layer, arena layout (`_v4_state_layout`), metadata |
| `atom/plugin/vllm/spec_decode_patch.py` | ATOM's patches to vLLM's spec-decode paths |
| `atom/models/deepseek_v4_dspark.py` | ATOM's DSpark draft (do not edit — compiled) |
| `atom/spec_decode/dspark_proposer.py` | native drafting loop; `_resolve_mtp_k` defines T |
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | the driver; defines the contract |
| `vllm/v1/worker/gpu/spec_decode/__init__.py` | `init_speculator` — the one selection point |
| `/app/ATOM/.claude/commands/add-atom-vllm-model.md` | the repo's own onboarding checklist — follow it |

Local scripts: `serve_dsv4_pro_atom_native_tp8.sh` (native baseline),
`serve_dsv4_pro_vllm_atom_tp8.sh` (plugin arm), `stop.sh`.

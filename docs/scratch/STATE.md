# Session state, 2026-09-02

Scratch branch. NOT for merge - working notes only, so the PR branch
(ds_v4_flash_0731_vllm_2_rebase) stays clean.

## aiter PR - ready to file, not yet filed

Branch: peizhang56/aiter @ ds_v4_flash_0731_vllm_2_rebase, commit 2eece445b
Base:   ROCm/aiter main @ d58537b70 (single commit, 2 files, +132/-1)
Title:  [Bugfix] Cache the MLA split count, not the split indptr tensor

Files: aiter/mla.py, op_tests/test_mla_split_indptr_cudagraph.py
DCO signed off. black/ruff clean. Test runs both as a plain script
(how .github/scripts/aiter_test.sh invokes op_tests) and under pytest:
exit 1 on unpatched main, exit 0 with the fix, 3/3 runs.

Drafts to paste into GitHub: aiter_issue.md, aiter_pr.md (this dir).

## What is proven vs not

PROVEN: get_meta_param returned an lru_cache'd CUDA tensor. A graph capturing
a cache hit records a bare pointer with no producing kernel; after eviction the
storage is recycled and replay reads garbage. Reproduced on MI355X, 3/3.
ATOM reaches it: paged_decode.py:1035 -> mla_decode_fwd_v4_nm with
num_kv_splits=None, and mla.py:1773's .to(device,dtype) is a no-op that returns
the SAME object (verified), so no defensive copy.

NOT PROVEN: that this causes the TP8 Memory Access Fault. Inferred only from
the address decode 0x7800000 = 480*8*16*512. The same signature still
reproduces at TP1 with the fix applied. Both drafts now state this as an open
lead, not a root cause.

OPEN EXPERIMENT (not run): instrument a real serving run - log the captured
indptr's data_ptr() at capture, then after N decode steps check whether that
cache entry still exists and still owns that storage. Settles whether the
hazard is reachable in production or only under the test's forced churn.

## Open decisions for the user

1. PR body template: aiter's CONTRIBUTE.md:446 defines its own
   (Summary/Motivation/Changes/Performance/Testing/Documentation/Dependencies/
   Breaking Changes), which differs from the aiter_pr_template.txt supplied.
   There is no .github/PULL_REQUEST_TEMPLATE.md, so neither auto-populates.
   Current drafts follow aiter_pr_template.txt.
2. Nothing is filed on ROCm/aiter yet - no issue, no PR.

## ATOM PR - blocked, do not file yet

Branch ds_v4_flash_0731_vllm_2_rebase @ ff77e9fb, 8 commits ahead of
ROCm/ATOM main, pushed to peizhang56/ATOM. Wait for the aiter merge first.

Blocking review items, unresolved:
- dspark_draft_kv_patch.py docstring says "UNMEASURED as of 2026-08-19:
  written against the source, never run"
- find_longest_cache_hit returns null_block for the hit region; safety rests on
  the unenforced invariant max(win_with_spec, index_topk) = 512 >= 128
- the branch rewrites DeepseekV4Model.forward, a @support_torch_compile file,
  which CLAUDE.md forbids - needs explicit justification or a rework

Non-blocking: hit_length alignment rounding can un-align (subtracting
% alignment_tokens then % block_size); oob_probe global mutation under Dynamo;
redundant _spec_cls/_registered globals; per-worker warn counters.
Body must call out the PR #2044 overlap on the DP idle-rank dummy batch.

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every `fp8_mqa_logits` call site must decide `clean_logits` on purpose.

The default is `clean_logits=True`, which fills the whole `[rows, total_committed]`
fp32 plane with `-inf` before the kernel runs -- a full-rectangle HBM pass (411 us
at `[1408, 350180]` on gfx950, bandwidth-bound) whose only consumer,
`top_k_per_row_prefill`, is handed the SAME `cu_starts`/`cu_ends` and never reads
outside a row's window. So on the paths where that holds the fill buys nothing,
and leaving the default in place is a silent ~0.4 ms/call tax rather than a
deliberate choice.

The decision is one keyword at a call site, which is exactly the kind of thing a
new backend copies from whichever sibling it was pasted from. This test is the
inventory: it reads the call sites out of the source, because reaching any of them
at runtime needs a GPU, a loaded model, and a chunked-prefill batch.

`tests/test_indexer_topk_row_window.py` and
`tests/test_indexer_logits_alignment_tail.py` pin the invariant this decision
rests on (aiter's top-k reads only `[rowStart, rowEnd)`, alignment padding
included); `tests/test_indexer_vllm_topk_row_window.py` pins the same for the
vLLM op that the vLLM plugin path uses instead.
"""

import ast
import pathlib

import pytest

ATOM_ROOT = pathlib.Path(__file__).resolve().parent.parent
ATOM_PKG = ATOM_ROOT / "atom"

# Call sites that still take the default `clean_logits=True`. Not a blessing --
# an inventory of unfinished work. The `-inf` fill there is believed redundant
# for the same reason it is redundant on the swept paths (each hands its top-k
# the same row bounds it gave the logits kernel), but that has not been measured
# end to end on those backends, so they are listed rather than assumed.
#
# Sweeping one => delete its line. Adding a NEW unswept call site => this test
# fails, which is the point: make the decision, do not inherit it.
KNOWN_UNSWEPT = {
    "atom/model_ops/glm5_next/indexer.py",
    "atom/plugin/rtpllm/attention_backend/rtp_sparse_mla_backend.py",
    "atom/plugin/sglang/attention_backend/sparse_mla_indexer.py",
}


def _call_sites():
    """`[(relpath, lineno, clean_logits_node_or_None)]` for every call in `atom/`."""
    sites = []
    for path in sorted(ATOM_PKG.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "fp8_mqa_logits(" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            )
            if name != "fp8_mqa_logits":
                continue
            kwarg = next(
                (k.value for k in node.keywords if k.arg == "clean_logits"), None
            )
            sites.append((str(path.relative_to(ATOM_ROOT)), node.lineno, kwarg))
    return sites


def test_the_scan_finds_the_call_sites_at_all():
    """The control: an import rename or a wrapper would make every assertion
    below vacuously true, and a test that silently stops testing is worse than
    no test."""
    sites = _call_sites()
    assert (
        len(sites) >= 4
    ), f"expected the known fp8_mqa_logits call sites, found {sites}"
    files = {relpath for relpath, _, _ in sites}
    assert "atom/plugin/vllm/attention/layer_sparse_mla.py" in files
    assert "atom/models/deepseek_v2.py" in files


def test_the_vllm_plugin_prefill_path_skips_the_inf_fill():
    """The vLLM plugin indexer scores `[rows, total_committed]` where
    `total_committed` is the sum of co-scheduled prefill contexts -- the widest
    buffer any of these call sites builds, so it is where the redundant fill
    costs the most. Its consumer is vLLM's `top_k_per_row_prefill`, bounded by
    the same `row_ks`/`row_ke` the logits kernel got."""
    sites = [
        (lineno, kwarg)
        for relpath, lineno, kwarg in _call_sites()
        if relpath == "atom/plugin/vllm/attention/layer_sparse_mla.py"
    ]
    assert len(sites) == 1, f"expected exactly one call site, found {sites}"
    lineno, kwarg = sites[0]
    assert kwarg is not None, (
        f"layer_sparse_mla.py:{lineno} fell back to clean_logits=True -- that is a "
        "full-rectangle -inf memset per indexer layer per prefill step, and nothing "
        "reads it"
    )
    assert isinstance(kwarg, ast.Constant) and kwarg.value is False, (
        f"layer_sparse_mla.py:{lineno} passes clean_logits={ast.dump(kwarg)}; this "
        "path requires the literal False"
    )


@pytest.mark.parametrize(
    "relpath",
    ["atom/models/deepseek_v2.py", "atom/models/deepseek_v4.py"],
)
def test_the_native_paths_stay_swept(relpath):
    """These two are what the plugin change was matched to. If one regresses to
    the default, the plugin's justification ('the native paths already do this')
    quietly stops being true."""
    kwargs = [kw for path, _, kw in _call_sites() if path == relpath]
    assert kwargs, f"no fp8_mqa_logits call site left in {relpath}"
    for kwarg in kwargs:
        assert (
            isinstance(kwarg, ast.Constant) and kwarg.value is False
        ), f"{relpath} no longer passes clean_logits=False"


def test_no_new_call_site_inherits_the_default():
    """A new backend that pastes the call without the keyword pays the fill
    forever and nobody notices, because the result is correct -- just slower.
    Fail on arrival instead."""
    unswept = {relpath for relpath, _, kwarg in _call_sites() if kwarg is None}
    new = unswept - KNOWN_UNSWEPT
    assert not new, (
        f"{sorted(new)} call fp8_mqa_logits without deciding clean_logits. Pass "
        "clean_logits=False if the only consumer is a per-row top-k bounded by the "
        "same cu_starts/cu_ends (see layer_sparse_mla.py), otherwise pass True "
        "explicitly and say why."
    )
    stale = KNOWN_UNSWEPT - unswept
    assert not stale, (
        f"{sorted(stale)} no longer takes the default -- drop it from KNOWN_UNSWEPT"
    )

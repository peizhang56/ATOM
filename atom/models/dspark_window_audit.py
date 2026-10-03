# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Arm-local audit of what the DSpark draft's rolling window actually delivers.

WHY. The ATOM-owned DSpark draft under the vLLM plugin accepts ~2.1 tokens/step
where config-matched native accepts ~3.50, and the whole of that gap is in the
window: ablating the context write collapses BOTH arms to the same ~1.05 floor,
so weights, LM head, Markov head, anchor, block width and the backbone stages
are all equivalent, and every *input* the draft consumes has been measured
equivalent too (``aux_concat`` and ``main_x`` at or below the same-arm noise
floor). What has never been measured is whether a window row the draft READS
holds the value that was WRITTEN for that position.

THE MEASUREMENT. ``_dspark_index_kernel`` declares a request's window to be
``n_valid = min(anchor + 1, W)`` rows ending at the anchor, and gathers them by
absolute position without checking that anything was ever stored there -- the
read side's own docstring says so ("anything left unwritten shows the slot's
previous occupant"). So ``n_valid`` is a CLAIM. This module measures the claim
against the fact:

* ``note_writes`` stamps ``stamp[slot, pos % ring_slots] = pos`` for exactly the
  rows ``write_context_kv`` just handed ``swa_write`` -- the last
  ``min(span, write_per_batch)`` rows of each request's span.
* ``note_reads`` takes the same ``(slot, anchor)`` pair the index kernel does,
  re-derives the ``n_valid`` window positions, and counts how many carry their
  own stamp.

``fresh / claimed`` is then the fraction of the window the draft reads that
holds real target KV for the position it believes it is reading. It is an
integer ratio with no numerical noise floor, it is computed inside one arm, and
it needs no comparison against the other arm to be meaningful: anything below
1.0 is the draft attending to another request's bytes.

ONE-SIDED. The stamp is the absolute position, so a previous occupant of the
same ring slot that wrote the SAME positions reads as fresh (unit-tested, case
2 in the module's own check). The audit therefore UNDER-reports staleness:
below 100% is a hard finding, 100% is not a full exoneration. Closing that
would need a per-request owner id, which neither arm exposes at this seam in
the same spelling.

This is the test `RESULTS.md` §7 lists as deferred four times in favour of
server A/Bs. It is deliberately NOT a cross-arm hidden-state diff -- two native
runs of one prompt land 14% apart on ``main_x`` while emitting identical text,
so nothing finer than ~15% is resolvable that way.

USE. ``ATOM_DSPARK_WINDOW_AUDIT=<report-every-N-steps>``; unset or 0 is off and
costs one attribute read per step. Both hook sites are eager on both arms
(``write_context_kv`` runs from the runner natively and from vLLM's speculator
in the plugin; ``dspark_attention`` is reached only through an opaque op), but
a CAPTURED draft pass replays without running either, so **run with eager /
piecewise cudagraphs** -- `kb/p1c` measured capture as acceptance-neutral
(2.376 vs 2.379), so eager is a faithful regime for this.

The device->host sync lives in ``maybe_report``, called from ``note_writes`` at
the top of a step, which is outside every captured region on both arms.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("atom")

# Starting height only -- the table GROWS. `state_slot_out` carries a PHYSICAL
# slot (`WindowParams.slot`: "a position in the plane, not a pool group"), which
# `UnifiedPoolGeometry.physical_slot` counts back from the top of the plane, so
# it is routinely far above `max_num_seqs` -- 1144-1155 was measured against
# `--max-num-seqs 512`. Clamping into a fixed table silently aliases distinct
# requests onto one stamp row and manufactures exactly the staleness this
# module exists to detect, so it must not be fixed-height.
_INITIAL_SLOTS = 2048

_STATE: _AuditState | None = None
_ENABLED: int | None = None


def audit_interval() -> int:
    """Steps between reports; 0 = disabled. Read once, then cached."""
    global _ENABLED
    if _ENABLED is None:
        try:
            _ENABLED = max(0, int(os.environ.get("ATOM_DSPARK_WINDOW_AUDIT", "0")))
        except ValueError:
            _ENABLED = 0
        if _ENABLED:
            logger.warning(
                "ATOM_DSPARK_WINDOW_AUDIT=%d: auditing DSpark window freshness "
                "every %d steps. Diagnostic only; run with eager or piecewise "
                "cudagraphs or the hooks never execute.",
                _ENABLED,
                _ENABLED,
            )
    return _ENABLED


class _AuditState:
    """Per-slot position stamps plus the running fresh/claimed tallies.

    Everything is a device tensor and stays one: the counters are read exactly
    once per report, from `maybe_report`, which runs at the top of a step.
    """

    def __init__(self, ring_slots: int, device: torch.device) -> None:
        self.ring_slots = int(ring_slots)
        self.num_slots = _INITIAL_SLOTS
        self.device = device
        # -1 is "never written". Positions are >= 0, so no real stamp collides.
        self.stamp = torch.full(
            (self.num_slots * self.ring_slots,), -1, dtype=torch.int64, device=device
        )
        self.max_slot_seen = 0
        # [fresh, claimed, steps, any] -- one tensor so a report is one sync.
        # `any` counts window rows that carry SOME stamp, whatever position it
        # names. It is what splits the two explanations for a stale row:
        #   any ~= claimed, fresh << claimed -> the slot holds another
        #       request's rows: a write/read slot or ordering mismatch.
        #   any ~= fresh  << claimed          -> those ring rows were never
        #       written at all: a coverage gap.
        self.tally = torch.zeros(4, dtype=torch.int64, device=device)
        self.last = torch.zeros(3, dtype=torch.int64, device=device)
        # Per-request detail for the worst step seen since the last report:
        # [slot, anchor, n_valid, fresh, any] x rows. Device-resident; read only
        # at report time.
        self.worst = torch.zeros(8, 5, dtype=torch.int64, device=device)
        self.worst_ratio = torch.ones(1, dtype=torch.float32, device=device)
        # The stale rows of the worst step's request 0, as
        # [offset-from-anchor, position wanted, position found]. A count is not
        # enough: 126/128 is harmless if the two are the window's oldest rows
        # and fatal if they are the two NEWEST, which are the ones the next
        # token actually depends on.
        self.worst_rows = torch.zeros(8, 3, dtype=torch.int64, device=device)
        self.steps_since_report = 0

    def ensure_slots(self, slots: torch.Tensor) -> None:
        """Grow the table so no physical slot has to be clamped into it.

        Costs one device->host sync per drafting step. Paid deliberately: an
        aliased stamp row is indistinguishable from the staleness being
        measured, so a cheap-but-clamping table would report a finding it
        manufactured. Both call sites are eager and outside every captured
        region on both arms, so the sync is legal here.
        """
        hi = int(slots.max().item()) if slots.numel() else 0
        if hi <= self.max_slot_seen:
            return
        self.max_slot_seen = hi
        if hi < self.num_slots:
            return
        new_slots = 2 * (hi + 1)
        grown = torch.full(
            (new_slots * self.ring_slots,), -1, dtype=torch.int64, device=self.device
        )
        grown[: self.stamp.numel()] = self.stamp
        logger.warning(
            "DSPARK WINDOW AUDIT: growing the stamp table %d -> %d physical "
            "slots (saw slot %d).",
            self.num_slots,
            new_slots,
            hi,
        )
        self.stamp = grown
        self.num_slots = new_slots

    def flat_index(self, slots: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """`(slot, pos) -> stamp row`. `ensure_slots` has made the clamp a no-op."""
        slot = slots.clamp(0, self.num_slots - 1)
        return slot * self.ring_slots + (pos % self.ring_slots)


def _state(ring_slots: int, device: torch.device) -> _AuditState:
    global _STATE
    if _STATE is None or _STATE.ring_slots != int(ring_slots):
        _STATE = _AuditState(ring_slots, device)
    return _STATE


@torch.no_grad()
def note_writes(
    window,  # WindowParams
    slots: torch.Tensor,  # [B] int, ring slot per request
    cu_seqlens_q: torch.Tensor,  # [B+1] int32, the TARGET batch's spans
    positions: torch.Tensor,  # [N] int64, absolute position per scheduled token
    write_per_batch: int,
) -> None:
    """Stamp the rows ``write_context_kv`` just wrote. Stage 0 only.

    Mirrors ``swa_write``'s own selection -- the LAST ``min(span,
    write_per_batch)`` rows of each request's span -- rather than assuming the
    whole span lands, because that bound is exactly what decides whether a long
    prefill chunk leaves the window covered.
    """
    if not audit_interval():
        return
    st = _state(window.ring_slots, positions.device)
    maybe_report(st)

    B = int(slots.numel())
    if B <= 0 or cu_seqlens_q.numel() < B + 1:
        return
    st.ensure_slots(slots)
    wpb = max(1, int(write_per_batch))
    cu = cu_seqlens_q.to(torch.int64)
    starts, ends = cu[:B], cu[1 : B + 1]
    n = torch.clamp(ends - starts, max=wpb)  # rows actually written
    first = ends - n
    j = torch.arange(wpb, device=positions.device, dtype=torch.int64)
    idx = first[:, None] + j[None, :]  # [B, wpb]
    live = j[None, :] < n[:, None]
    idx = idx.clamp_(0, max(0, positions.numel() - 1))
    pos = positions.reshape(-1)[idx]  # [B, wpb]
    flat = st.flat_index(slots.to(torch.int64)[:, None].expand_as(pos).reshape(-1),
                         pos.reshape(-1))
    keep = live.reshape(-1)
    st.stamp[flat[keep]] = pos.reshape(-1)[keep]


@torch.no_grad()
def note_reads(
    window,  # WindowParams
    slots: torch.Tensor,  # [B] int, ring slot per request
    anchors: torch.Tensor,  # [B] int64, anchor position per request
    draft_window: int,  # W
) -> None:
    """Count how much of the window the draft is about to read is really its own.

    Re-derives ``_dspark_index_kernel``'s own window span rather than reading
    ``kv_indices`` back: the point is to test the kernel's CLAIM, and taking the
    claim from the kernel's output would test only that the gather is
    self-consistent -- which the integer round-trip already showed it is.
    """
    if not audit_interval():
        return
    st = _state(window.ring_slots, anchors.device)
    B = int(anchors.numel())
    if B <= 0:
        return
    st.ensure_slots(slots)
    W = int(draft_window)
    anchor = anchors.to(torch.int64).reshape(-1)
    n_valid = torch.clamp(anchor + 1, max=W)  # the kernel's own formula
    j = torch.arange(W, device=anchor.device, dtype=torch.int64)
    inw = j[None, :] < n_valid[:, None]  # [B, W]
    pos = anchor[:, None] - n_valid[:, None] + 1 + j[None, :]
    slot_b = slots.to(torch.int64).reshape(-1)
    flat = st.flat_index(slot_b[:, None].expand_as(pos).reshape(-1),
                         pos.clamp(min=0).reshape(-1))
    stamped = st.stamp[flat].reshape(B, W)
    fresh = (stamped == pos) & inw
    any_row = (stamped >= 0) & inw
    st.last[0] = fresh.sum()
    st.last[1] = inw.sum()
    st.last[2] = any_row.sum()
    st.tally[0] += st.last[0]
    st.tally[1] += st.last[1]
    st.tally[2] += 1
    st.tally[3] += st.last[2]
    st.steps_since_report += 1

    # Keep the per-request detail of the worst step in this report window. The
    # comparison is on device -- `torch.where`, not a Python `if` -- so no step
    # pays a sync for it.
    ratio = st.last[0].to(torch.float32) / st.last[1].clamp(min=1).to(torch.float32)
    take = ratio < st.worst_ratio
    rows = torch.zeros_like(st.worst)
    k = min(B, st.worst.shape[0])
    rows[:k, 0] = slot_b[:k]
    rows[:k, 1] = anchor[:k]
    rows[:k, 2] = n_valid[:k]
    rows[:k, 3] = fresh[:k].sum(dim=1)
    rows[:k, 4] = any_row[:k].sum(dim=1)
    st.worst = torch.where(take, rows, st.worst)

    # Request 0's stale rows, newest-first: `anchor - pos` is 0 for the anchor
    # itself, so a small offset means a row the draft critically depends on.
    stale = inw[0] & ~fresh[0]
    order = torch.argsort((anchor[0] - pos[0]) + torch.where(stale, 0, 1 << 40))
    pick = order[: st.worst_rows.shape[0]]
    detail = torch.stack(
        [anchor[0] - pos[0][pick], pos[0][pick], stamped[0][pick]], dim=1
    )
    detail = torch.where(stale[pick].reshape(-1, 1), detail, torch.full_like(detail, -1))
    st.worst_rows = torch.where(take, detail, st.worst_rows)
    st.worst_ratio = torch.where(take, ratio.reshape(1), st.worst_ratio)


def maybe_report(st: _AuditState) -> None:
    """Print, at most every ``audit_interval()`` steps. The one sync point.

    Called from ``note_writes`` -- the top of a step, and eager on both arms --
    so the counters it reads were finished by the previous step's draft pass.
    """
    every = audit_interval()
    if not every or st.steps_since_report < every:
        return
    st.steps_since_report = 0
    fresh, claimed, steps, any_rows = (int(v) for v in st.tally.tolist())
    if claimed <= 0:
        return
    worst = [[int(v) for v in row] for row in st.worst.tolist() if int(row[2]) > 0]
    stale = [[int(v) for v in row] for row in st.worst_rows.tolist() if int(row[1]) >= 0]
    st.worst_ratio.fill_(1.0)
    logger.warning(
        "DSPARK WINDOW AUDIT: cumulative fresh=%d/%d (%.1f%%), any-stamp %d/%d "
        "(%.1f%%) over %d drafting steps. 'fresh' = window rows whose stamp is "
        "the position the draft reads them as; 'any-stamp' = rows ever written "
        "at all in that slot. fresh<<any => the slot holds another request's "
        "rows; any<<claimed => those rows were never written.\n"
        "  worst step in this window, per request "
        "(slot, anchor, n_valid, fresh, any): %s\n"
        "  its request 0's stale rows, newest first "
        "(anchor-pos, pos wanted, pos found): %s",
        fresh,
        claimed,
        100.0 * fresh / claimed,
        any_rows,
        claimed,
        100.0 * any_rows / claimed,
        steps,
        worst,
        stale,
    )

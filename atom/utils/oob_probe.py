"""Count out-of-range token ids reaching embedding gathers. Off by default.

Enable with ``ATOM_OOB_PROBE=1``. Answers one question: how often does the
async-spec-decode ``-1`` placeholder actually reach a gather, and does it
differ between ``--async-scheduling`` on and off? That decides whether the
bounds checks in 78a5d0aa are inert insurance or are silently costing
acceptance, which is the metric the DSpark work is trying to raise.

Counting must not perturb what it measures. The comparison and the accumulate
are GPU-side, so no host sync is introduced on the hot path; the counters are
only read back every ``ATOM_OOB_PROBE_EVERY`` calls (default 500), which is
rare enough to leave the CPU/GPU run-ahead that produces the placeholder
intact. A ``.item()`` on every call would drain the queue and hide it.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("ATOM_OOB_PROBE", "") == "1"
_EVERY = int(os.environ.get("ATOM_OOB_PROBE_EVERY", "500"))

# tag -> [oob_count, total_count] as a single int64 GPU tensor, plus a call
# counter kept on the host (never a sync -- it is a plain Python int).
_counters: dict[str, torch.Tensor] = {}
_calls: dict[str, int] = {}


def count(tag: str, ids: torch.Tensor, lo: int, hi: int) -> None:
    """Accumulate how many of `ids` fall outside [lo, hi). No host sync."""
    if not ENABLED:
        return
    try:
        c = _counters.get(tag)
        if c is None:
            c = torch.zeros(2, dtype=torch.int64, device=ids.device)
            _counters[tag] = c
            _calls[tag] = 0
        oob = ((ids < lo) | (ids >= hi)).sum()
        c[0] += oob
        c[1] += ids.numel()
        _calls[tag] += 1
        if _calls[tag] % _EVERY == 0:
            n_oob, n_tot = (int(v) for v in c.tolist())  # the only sync
            logger.info(
                "OOB probe [%s]: %d / %d ids out of [%d, %d) over %d calls "
                "(%.4f%% of ids; range floor was %s)",
                tag, n_oob, n_tot, lo, hi, _calls[tag],
                100.0 * n_oob / max(1, n_tot),
                int(ids.min()) if n_oob else "n/a",
            )
    except Exception as e:  # a probe must never take the server down
        logger.warning("OOB probe [%s] disabled after error: %s", tag, e)
        globals()["ENABLED"] = False

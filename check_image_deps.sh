#!/usr/bin/env bash
# Is the CONTAINER IMAGE broken, or is it this node?
#
# Written 2026-09-25 after `serve_glm53_mxfp4_attnfp8_mtp_tp8_ep.sh` died at
# import time on rocm/atom-dev:vllm-latest with
#
#   TypeError: load_lib_module() got an unexpected keyword argument 'extra_lib_paths'
#
# raised from xgrammar/load_binding.py into tvm_ffi.libinfo. vLLM imports
# xgrammar eagerly (vllm/v1/request.py -> structured_output/backend_xgrammar.py,
# at class-body time), so this kills `vllm serve` before a single weight is
# loaded and NO serve flag routes around it. The same node had swept cleanly at
# 11:10 the same morning.
#
# The suspicion this script exists to settle: the image shipped a package set
# that no resolver would produce -- xgrammar 0.2.8 requires
# apache-tvm-ffi>=0.1.11 while vLLM pins apache-tvm-ffi==0.1.10 exactly, which
# are mutually unsatisfiable. If that is baked into the image, every container
# from it fails identically and rolling back the image is the fix; if it is not,
# something mutated this particular container after it started.
#
# USAGE -- run it once in every new container, before the first server start:
#
#   ./check_image_deps.sh
#
# It needs no GPU, no weights and no network, and it does not import torch, so
# it is seconds rather than minutes.
#
# IT DIAGNOSES FIRST, THEN REPAIRS -- and only the one conflict above, only when
# the live import actually fails. A clean container is left untouched: the
# report is still the point, and an unconditional pip install would mutate an
# image that was fine and hide what it shipped.
#
# The repair is `pip install --no-deps "xgrammar==0.2.7"`. Fixed from the
# XGRAMMAR side deliberately: upgrading apache-tvm-ffi to 0.1.11 also satisfies
# xgrammar 0.2.8 and violates vLLM's exact pin, which is the one that must not
# move. 0.2.7 declares >=0.1.10 and is satisfied by what is installed, so the
# set becomes consistent rather than merely working.
#
# THE REPAIR IS CONTAINER-LOCAL AND DIES WITH THE CONTAINER. That is why this
# runs per container rather than once per node, and why the real fix is still
# pinning a compatible xgrammar in the image build.
#
# Exit status: 0 = the import that crashed now succeeds (whether it already did
# or this script fixed it); 1 = still broken, and the report says what lost.

set -uo pipefail

PY=/opt/venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
[ -x "$PY" ] || { echo "no python3 found -- is this the right image?" >&2; exit 1; }

echo "==============================================================="
echo " image dependency check -- $(date -Is)"
echo "==============================================================="

# ---------------------------------------------------------------- identity
# Which image/container is even answering. `docker inspect` is not available
# from inside, so the identity has to be reconstructed from what the container
# can see: the cgroup/hostname gives the container id, and the mtime of
# site-packages is the closest thing to a build timestamp -- it is what
# separated the working 11:10 sweep from the 19:19 crash on this node.
echo
echo "--- identity ---"
echo "hostname (container id, usually): $(hostname)"
[ -r /etc/os-release ] && echo "os:        $(. /etc/os-release; echo "$PRETTY_NAME")"
for f in /opt/rocm/.info/version /opt/rocm/.info/version-dev; do
    [ -r "$f" ] && echo "rocm:      $(cat "$f")"
done
echo "python:    $("$PY" -V 2>&1) ($PY)"

SP=$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
echo "site-pkgs: $SP"
if [ -n "$SP" ] && [ -d "$SP" ]; then
    echo "  mtime of site-packages entries (a proxy for image build time):"
    for p in vllm xgrammar tvm_ffi tilelang; do
        [ -e "$SP/$p" ] && printf "    %-10s %s\n" "$p" \
            "$(stat -c %y "$SP/$p" 2>/dev/null | cut -d. -f1)"
    done
fi

# ------------------------------------------------------------- the conflict
# The heart of it. For every installed distribution, pull its declared
# Requires-Dist on apache-tvm-ffi and test the INSTALLED version against that
# specifier. A violated constraint here is a packaging fact -- it does not
# depend on hardware, driver, weights or anything else about the node, which is
# exactly why it settles "image or node?".
echo
echo "--- versions ---"
"$PY" - <<'PY'
import importlib.metadata as md
for name in ("vllm", "xgrammar", "apache-tvm-ffi", "tilelang", "torch", "aiter"):
    try:
        print(f"  {name:16s} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:16s} (not installed)")
PY

echo
echo "--- declared constraints on apache-tvm-ffi vs what is installed ---"
"$PY" - <<'PY'
import importlib.metadata as md
import sys

try:
    from packaging.requirements import Requirement
except ImportError:
    print("  packaging not available; cannot evaluate specifiers")
    sys.exit(0)

TARGET = "apache-tvm-ffi"
try:
    installed = md.version(TARGET)
except md.PackageNotFoundError:
    installed = None
print(f"  installed {TARGET}: {installed or '(absent)'}")
print()

violations = []
for dist in sorted(md.distributions(), key=lambda d: d.metadata["Name"] or ""):
    name = dist.metadata["Name"]
    for raw in dist.requires or ():
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.name.lower().replace("_", "-") != TARGET:
            continue
        # Skip requirements gated behind an extra we did not install.
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        if installed is None:
            ok = False
        else:
            # prereleases=True so a .postN/.devN installed build is judged on
            # its version, not silently excluded by the specifier machinery.
            ok = req.specifier.contains(installed, prereleases=True)
        flag = "ok     " if ok else "VIOLATED"
        print(f"  {flag}  {name:12s} requires {TARGET}{req.specifier}")
        if not ok:
            violations.append((name, str(req.specifier)))

print()
if violations:
    print("  => UNSATISFIABLE AS SHIPPED. No dependency resolver produces this")
    print("     set; it can only come from an image built with --no-deps, a")
    print("     layer installed after the solve, or a hand edit.")
else:
    print("  => all declared apache-tvm-ffi constraints are satisfied.")
sys.exit(1 if violations else 0)
PY
CONFLICT_RC=$?

# --------------------------------------------------------------- pip check
# Broader than the hand-rolled check above: catches any OTHER dependency the
# image left inconsistent, not just the one that bit us.
#
# INFORMATIONAL ONLY -- deliberately NOT part of the verdict. These ROCm images
# ship a long-standing tail of unrelated complaints (lmcache wanting awscrt and
# cufile-python, a triton pin torch disagrees with, numpy above lmcache's cap),
# all of which were equally true on the image that swept cleanly at 11:10.
# Gating the verdict on `pip check` would therefore fail EVERY image, including
# a good one, and drown the one constraint that actually matters. Diff this
# section between two images to spot what changed; do not read it as pass/fail.
echo
echo "--- pip check (whole environment; informational, not the verdict) ---"
PIPOUT=$("$PY" -m pip check 2>&1)
PIPRC=$?
echo "$PIPOUT" | sed 's/^/  /'
[ "$PIPRC" -ne 0 ] && echo "  (pre-existing noise unless it names xgrammar/tvm-ffi/vllm)"

# ------------------------------------------------------- the live reproducer
# The authority. Everything above is metadata; this executes the exact import
# chain that killed the server. Run in a subprocess so a hard crash is caught
# rather than taking the report with it. torch is not imported -- only the
# binding load that actually fails -- so it stays fast.

# A function, not a straight-line block, because the repair below has to ask
# the same question again afterwards. Re-running the real import is the only
# way to know the repair took -- pip reporting success says a wheel landed, not
# that the binding loads.
live_import() {
    IMPORT_ERR=$("$PY" - <<'PY' 2>&1 >/dev/null
import xgrammar                                            # loads the native binding
from xgrammar.load_binding import LIB                       # the exact failure site
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
PY
)
    IMPORT_RC=$?
    if [ "$IMPORT_RC" -eq 0 ]; then
        echo "  OK -- xgrammar binding loads and vLLM's xgrammar backend imports."
    else
        echo "  FAILED (exit $IMPORT_RC). Tail of the traceback:"
        echo "$IMPORT_ERR" | grep -vE "^(INFO|WARNING|\[atom)" | tail -12 | sed 's/^/    /'
    fi
    return "$IMPORT_RC"
}

echo
echo "--- live import reproducer ---"
live_import
IMPORT_RC=$?

# ------------------------------------------------------------------ repair
# Only when the import actually failed. A container that works is left exactly
# as it was -- see the header: an unconditional install would mutate a good
# image and destroy the evidence of what it shipped.
REPAIRED=0
if [ "$IMPORT_RC" -ne 0 ]; then
    echo
    echo "--- repair: pip install --no-deps xgrammar==0.2.7 ---"
    if "$PY" -m pip install --no-deps "xgrammar==0.2.7" 2>&1 | sed 's/^/  /'; then
        echo
        echo "  re-running the live import reproducer:"
        live_import
        IMPORT_RC=$?
        [ "$IMPORT_RC" -eq 0 ] && REPAIRED=1
        # The constraint check ran against the pre-repair environment, so its
        # verdict is stale now. Recompute it rather than carry the old one into
        # the verdict below.
        if [ "$IMPORT_RC" -eq 0 ]; then
            "$PY" -c 'import importlib.metadata as m
from packaging.requirements import Requirement
i = m.version("apache-tvm-ffi")
bad = [d.metadata["Name"] for d in m.distributions() for r in (d.requires or ())
       if (lambda q: q.name.lower().replace("_","-") == "apache-tvm-ffi"
           and not q.specifier.contains(i, prereleases=True))(Requirement(r))]
raise SystemExit(1 if bad else 0)' 2>/dev/null
            CONFLICT_RC=$?
        fi
    else
        echo "  pip install FAILED -- no network, or no 0.2.7 wheel for this python."
    fi
fi

# ----------------------------------------------------------------- verdict
#
# The verdict rests on exactly two things: the live import (the authority --
# it either gets past the line that killed the server or it does not) and the
# apache-tvm-ffi constraint check (which explains WHY, and catches a set that
# happens to import today but is still unshippable). Everything else above is
# context for diffing two images.
#
# Exit code answers one question: will `vllm serve` get past import in THIS
# container. Repaired counts as yes -- that is the point of repairing.
echo
echo "==============================================================="
if [ "$IMPORT_RC" -ne 0 ]; then
    echo " VERDICT: this container is BROKEN for vllm serve, and the repair"
    echo "   did not take. It dies at import, before weights load -- no GPU or"
    echo "   node state involved. Roll the image back, or pin a compatible"
    echo "   xgrammar in the build."
    VERDICT=1
elif [ "$REPAIRED" -eq 1 ]; then
    echo " VERDICT: was BROKEN, now REPAIRED. vllm serve will start here."
    echo "   The image itself is still broken -- this patch is container-local"
    echo "   and dies with the container. Re-run this script in the next one."
    VERDICT=0
elif [ "$CONFLICT_RC" -eq 0 ]; then
    echo " VERDICT: this image is CLEAN. Nothing was installed."
    echo "   vllm serve will get past import here."
    VERDICT=0
else
    echo " VERDICT: imports, but the dependency set is INCONSISTENT."
    echo "   vllm serve will start, so nothing was installed -- but something"
    echo "   above violates a declared pin and it works by luck rather than by"
    echo "   construction. Treat the image as unshippable."
    VERDICT=0
fi
echo "==============================================================="

exit "$VERDICT"

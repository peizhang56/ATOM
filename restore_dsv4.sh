#!/usr/bin/env bash
# Put /app/aiter-test, /app/ATOM and /app/vllm back on the DeepSeek-V4-Pro
# arm's branch.
#
# aiter and ATOM are `pip install -e`'d (aiter -> /app/aiter-test/aiter,
# atom -> /app/ATOM/atom), so the working tree IS what the next server start
# imports -- CLAUDE.md's "an uncommitted edit is already in the next
# measurement". That cuts both ways: a stray checkout, a half-applied patch or
# a container that came up on ROCm/main silently changes the arm under test
# while arm.json still looks plausible. This script restores the set to a
# known point and then prints what it actually left behind.
#
# /app/vllm IS NOT LIKE THE OTHER TWO. The runtime copy is
# /opt/venv/lib/python3.12/site-packages/vllm, a real directory from a
# non-editable install -- editing or checking out /app/vllm changes NOTHING
# about what the next server imports (CLAUDE.md §2.1). It is restored here
# because it is the source of truth for *reading* vLLM, because arm.json
# records its sha, and because a container that came up detached makes that
# sha unattributable. If vLLM's own code needs to change at runtime, the
# source tree is not the lever -- rebuild/reinstall, or patch site-packages.
#
# WHAT IT DOES, per repo:
#   1. `upstream` -> the fork that carries the branch (added, or re-pointed if
#      it exists with a different URL). `origin` is left alone -- it is still
#      ROCm/aiter, ROCm/ATOM and vllm-project/vllm, and the shas in arm.json
#      are read against it.
#   2. fetch just refs/heads/$BRANCH from upstream.
#   3. `git checkout -B $BRANCH upstream/$BRANCH` -- so an existing local
#      branch is RESET to the fork, not merged or rebased onto it.
#   4. submodules, if the branch has any (aiter: 3rdparty/composable_kernel).
#
# USAGE
#   ./restore_dsv4.sh                  # the normal case
#   BRANCH=some_other ./restore_dsv4.sh
#   FORCE=1 ./restore_dsv4.sh          # discard local modifications, and run
#                                      # even with a server up (see below)
#   CLEAN_JIT=1 ./restore_dsv4.sh      # also delete aiter's prebuilt .so files
#   VLLM_FORCE=0 ./restore_dsv4.sh     # make vllm refuse on a dirty tree too
#
# IT REFUSES on a dirty tracked tree, and it refuses while a server is running.
# Neither is bureaucracy:
#
#   - A dirty tree is the one state where the measurement and the sha disagree
#     on purpose. Blowing it away loses work that no commit holds. Stash it, or
#     say FORCE=1 to mean it.
#
#     vllm is the standing exception (VLLM_FORCE, default 1). Its container
#     image ships detached at the branch's parent with the branch's own diff
#     uncommitted -- as of 2026-10-03, `git diff upstream/ds_v4_atom_vllm` over
#     that tree is empty, i.e. the "modifications" ARE the target commit
#     (281cfd5a09 "Add dirty file from vllm", the ROCm 7.2 / torch 2.10 build
#     patch, also on the fork as build/rocm7.2-torch2.10-compat). Refusing
#     there would make the common case require FORCE=1, which would then also
#     silence the refusal for aiter and ATOM -- where it is load-bearing. The
#     script prints the diff before discarding, and that diff being non-empty
#     is the signal that the image changed under you.
#   - Under a live server, the engine has already imported some of these
#     modules and has NOT imported others (aiter JIT-loads kernels lazily, on
#     first call). Swapping the source tree mid-run gives a process that is
#     half one branch and half the other, and the numbers it produces are
#     attributable to neither. `./stop.sh` first.
#
# IT DOES NOT touch aiter's JIT artifacts by default, and that is the one sharp
# edge left. aiter/jit/core.py:1675 reuses an existing `<module>.so` whenever
# the file exists and its offload arch matches the running GPU -- it does not
# hash the sources it was built from. So kernels built on the old branch keep
# being loaded after this script moves the source tree. ~5 GB of
# aiter/jit/build is also what makes the first run after a wipe take tens of
# minutes, which is why wiping is not the default.
#
# The script compares the old and new HEAD over the kernel source paths and
# says so loudly when they differ. The fixes, cheapest first:
#     AITER_REBUILD=2 ./serve_...        # drop the .so, keep the build cache
#     AITER_REBUILD=1 ./serve_...        # drop both -- a full cold rebuild
#     CLEAN_JIT=1 ./restore_dsv4.sh      # same as level 2, done here and now
#
# Exit status: 0 = both repos are on $BRANCH at the fork's tip; 1 = at least
# one is not, and the reason is on stderr. Nothing is left half-restored
# silently -- the summary at the end reports each repo's real state.

set -uo pipefail

BRANCH=${BRANCH:-ds_v4_atom_vllm}
FORCE=${FORCE:-0}
CLEAN_JIT=${CLEAN_JIT:-0}
# Per-repo override of the dirty-tree refusal; see the header. 1 = reset the
# tree and check out anyway, without loosening anything for aiter/ATOM.
VLLM_FORCE=${VLLM_FORCE:-1}

AITER_DIR=${AITER_DIR:-/app/aiter-test}
ATOM_DIR=${ATOM_DIR:-/app/ATOM}
VLLM_DIR=${VLLM_DIR:-/app/vllm}
AITER_UPSTREAM=${AITER_UPSTREAM:-git@github.com:peizhang56/aiter.git}
ATOM_UPSTREAM=${ATOM_UPSTREAM:-git@github.com:peizhang56/ATOM.git}
VLLM_UPSTREAM=${VLLM_UPSTREAM:-git@github.com:peizhang56/vllm.git}
# Where the vLLM that actually runs lives. Not a symlink to VLLM_DIR and not an
# editable install -- used only to warn when the two have drifted.
VLLM_SITE=${VLLM_SITE:-/opt/venv/lib/python3.12/site-packages}

# The forks are private, so this is ssh with /root/.ssh/id_rsa, not https.
# accept-new rather than the default `ask`: a fresh container has no
# known_hosts at all, and the default turns that into "Host key verification
# failed" on a non-tty -- which is how this fails in a hook or a one-liner.
# accept-new still refuses a host key that CHANGED after it was recorded, which
# is the case that matters; it only trusts the first sighting.
export GIT_SSH_COMMAND=${GIT_SSH_COMMAND:-"ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"}

# Paths whose contents decide what aiter's JIT compiles. Kept narrow on
# purpose: a diff confined to aiter/*.py changes what runs immediately (pip -e)
# and needs no rebuild, so listing those here would cry wolf on every restore.
AITER_KERNEL_PATHS=(csrc hsa 3rdparty aiter/jit)

RC=0
declare -a SUMMARY=()
declare -a NOTES=()

say()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }
die()  { printf '%s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
# A server holding these modules open is checked once, before either repo is
# touched -- restoring aiter and then bailing on ATOM would leave the pair
# mismatched, which is worse than not starting.
echo "==============================================================="
echo " restore_dsv4 -- $(date -Is)"
echo "==============================================================="
say
say "branch:   $BRANCH"
say "aiter:    $AITER_DIR  <- $AITER_UPSTREAM"
say "ATOM:     $ATOM_DIR  <- $ATOM_UPSTREAM"
say "vllm:     $VLLM_DIR  <- $VLLM_UPSTREAM  (source only; runtime is $VLLM_SITE/vllm)"

# Same pattern stop.sh reaps, minus the aiperf half: a client alone imports
# none of this and is no reason to refuse.
#
# Our own ancestry is excluded by PID for stop.sh's reason: `pgrep -f` matches
# the whole command line, so a launcher shell whose argv happens to mention
# atom would otherwise look like a live server and refuse every restore. Only a
# refusal here rather than a SIGKILL, but a false one is still a wall.
ancestry() {
  local p=$$
  while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    echo "$p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
}
SERVER_PIDS=$(pgrep -f "vllm serve|VLLM::|ATOM::|atom.entrypoints|openai_server" 2>/dev/null \
              | sort -u | comm -23 - <(ancestry | sort -u) | tr '\n' ' ')
if [ -n "${SERVER_PIDS// /}" ]; then
    say
    if [ "$FORCE" = "1" ]; then
        warn "WARNING: a server is still up (pids: $SERVER_PIDS) and FORCE=1 says go."
        warn "         Anything it measures from here is half one branch, half the"
        warn "         other. Do not report numbers from that process."
    else
        warn "REFUSING: a server is still running (pids: $SERVER_PIDS)."
        warn "  It has already imported part of these trees and will lazily import"
        warn "  the rest from whatever this script leaves on disk. Run ./stop.sh"
        warn "  first (it also waits for VRAM to drain), or FORCE=1 to override."
        exit 1
    fi
fi

# ------------------------------------------------------------------ restore
# $1 label, $2 dir, $3 upstream url, $4 force-on-dirty (optional, default FORCE)
restore_repo() {
    local label=$1 dir=$2 url=$3 force=${4:-$FORCE}
    say
    say "--------------------------------------------------------------"
    say " $label -- $dir"
    say "--------------------------------------------------------------"

    [ -d "$dir/.git" ] || { warn "  not a git repo: $dir"; RC=1
        SUMMARY+=("$label|MISSING|-|-|-"); return 1; }

    local old_head old_desc
    old_head=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
    old_desc=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null)
    say "  currently: $old_desc @ ${old_head:0:9}"

    # Dirty = tracked modifications only. Untracked files are build output
    # here (aiter/jit/flydsl_cache, *.so) and git checkout carries them across
    # branches untouched, so they are not a reason to stop.
    #
    # Only REPORTED here; the decision is deferred until after the fetch, so it
    # can be made against the branch we are about to check out rather than in
    # the dark. The fetch is read-only, so nothing is lost by paying it first.
    local dirty
    dirty=$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null)
    if [ -n "$dirty" ]; then
        say "  tracked modifications:"
        printf '%s\n' "$dirty" | sed 's/^/    /' | head -20
    fi

    # Remote: add, or re-point if something else already claims the name.
    local have
    if have=$(git -C "$dir" remote get-url upstream 2>/dev/null); then
        if [ "$have" = "$url" ]; then
            say "  remote upstream: already $url"
        else
            say "  remote upstream: $have -> $url"
            git -C "$dir" remote set-url upstream "$url" || { warn "  set-url failed"; RC=1
                SUMMARY+=("$label|FAILED (remote)|$old_desc|${old_head:0:9}|-"); return 1; }
        fi
    else
        say "  remote upstream: adding $url"
        git -C "$dir" remote add upstream "$url" || { warn "  remote add failed"; RC=1
            SUMMARY+=("$label|FAILED (remote)|$old_desc|${old_head:0:9}|-"); return 1; }
    fi

    # One branch, not the whole fork. `git remote add` already wrote the
    # +refs/heads/*:refs/remotes/upstream/* refspec for anyone who later wants
    # the rest; this narrow fetch is what keeps a restore to seconds.
    say "  fetching upstream/$BRANCH ..."
    if ! git -C "$dir" fetch --quiet upstream \
            "+refs/heads/$BRANCH:refs/remotes/upstream/$BRANCH" 2>&1 | sed 's/^/    /'; then
        warn "  fetch failed -- no such branch on the fork, or ssh could not"
        warn "  authenticate (key: ~/.ssh/id_rsa). Try:"
        warn "    GIT_SSH_COMMAND=\"\$GIT_SSH_COMMAND -v\" git -C $dir fetch upstream"
        RC=1
        SUMMARY+=("$label|FAILED (fetch)|$old_desc|${old_head:0:9}|-")
        return 1
    fi

    local target
    target=$(git -C "$dir" rev-parse "upstream/$BRANCH" 2>/dev/null) \
        || { warn "  upstream/$BRANCH did not resolve after fetch"; RC=1
             SUMMARY+=("$label|FAILED (fetch)|$old_desc|${old_head:0:9}|-"); return 1; }
    say "  upstream/$BRANCH = ${target:0:9}"

    # Now the dirty decision, with the target in hand. The interesting case is
    # a tree whose modifications are ALREADY the target's content -- a
    # container shipped detached at the branch's parent with the branch's diff
    # applied but not committed. Discarding that is provably a no-op, and
    # saying so is the difference between a scary warning and a true one.
    if [ -n "$dirty" ]; then
        local same_as_target=0
        git -C "$dir" diff --quiet "upstream/$BRANCH" -- 2>/dev/null && same_as_target=1
        if [ "$same_as_target" = "1" ]; then
            say "  ...which are byte-identical to upstream/$BRANCH. Discarding them"
            say "     changes no file content -- it only moves HEAD onto the branch."
        elif [ "$force" != "1" ]; then
            warn "  REFUSING to discard them: they are NOT in upstream/$BRANCH."
            warn "  \`git -C $dir stash\` to keep them, or re-run with FORCE=1."
            RC=1
            SUMMARY+=("$label|REFUSED (dirty)|$old_desc|${old_head:0:9}|dirty")
            return 1
        else
            warn "  force: discarding modifications that are NOT in upstream/$BRANCH."
            warn "         \`git -C $dir stash list\` is empty -- they are saved nowhere."
            git -C "$dir" diff --stat | sed 's/^/           /' | tail -20 >&2
            NOTES+=("$label: discarded local edits not present in upstream/$BRANCH")
        fi
        git -C "$dir" reset --hard HEAD >/dev/null || { warn "  reset failed"; RC=1
            SUMMARY+=("$label|FAILED (reset)|$old_desc|${old_head:0:9}|dirty"); return 1; }
    fi

    # -B, so an existing local $BRANCH is reset to the fork rather than merged.
    # Divergent local commits on it are dropped from the branch -- still in the
    # reflog, and the old sha is printed below precisely so they are reachable.
    if ! git -C "$dir" checkout -B "$BRANCH" "upstream/$BRANCH" 2>&1 | sed 's/^/    /'; then
        warn "  checkout failed"
        RC=1
        SUMMARY+=("$label|FAILED (checkout)|$old_desc|${old_head:0:9}|-")
        return 1
    fi

    if [ -f "$dir/.gitmodules" ]; then
        say "  submodules ..."
        git -C "$dir" submodule update --init --recursive 2>&1 | sed 's/^/    /'
    fi

    local new_head new_dirty
    new_head=$(git -C "$dir" rev-parse HEAD)
    new_dirty=$(git -C "$dir" status --porcelain --untracked-files=no)
    say "  now: $BRANCH @ ${new_head:0:9}$([ -n "$new_dirty" ] && echo ' (DIRTY)')"
    if [ "$old_head" != "$new_head" ]; then
        say "  previous HEAD ${old_head:0:9} is still reachable:"
        say "    git -C $dir checkout $old_head"
    fi

    SUMMARY+=("$label|ok|$BRANCH|${new_head:0:9}|$([ -n "$new_dirty" ] && echo dirty || echo clean)")

    # --- the stale-kernel question, aiter only ---------------------------
    if [ "$label" = "aiter" ] && [ -n "$old_head" ] && [ "$old_head" != "$new_head" ]; then
        if ! git -C "$dir" diff --quiet "$old_head" "$new_head" -- "${AITER_KERNEL_PATHS[@]}" 2>/dev/null; then
            NOTES+=("aiter kernel sources changed between ${old_head:0:9} and ${new_head:0:9}")
            warn
            warn "  NOTE: ${AITER_KERNEL_PATHS[*]} differ across that checkout, and the"
            warn "        prebuilt .so files in aiter/jit were compiled from the OLD"
            warn "        ones. aiter reuses a .so whenever it exists (jit/core.py:1675)"
            warn "        -- it never checks the sources. Next server start should be"
            warn "        AITER_REBUILD=2, or re-run this with CLEAN_JIT=1."
        fi
    fi
}

restore_repo aiter "$AITER_DIR" "$AITER_UPSTREAM"
restore_repo ATOM  "$ATOM_DIR"  "$ATOM_UPSTREAM"
restore_repo vllm  "$VLLM_DIR"  "$VLLM_UPSTREAM" "$VLLM_FORCE"

# --------------------------------------------------- vllm source vs runtime
# The one check that earns its keep for a non-editable repo: after moving the
# source tree, does the code the engine will actually import still match it?
#
# NOT answered from shas. The dist-info records the commit the wheel was built
# at (0.28.1.dev0+g2cf0a6915.d20261003.rocm724 -> 2cf0a6915), and on this image
# that is the PARENT of ds_v4_atom_vllm: the build ran with the branch's diff
# applied but uncommitted, so the label is one commit stale while the installed
# .py files are the branch's exactly. A sha comparison calls that a drift and
# is wrong. Compare the files.
#
# Only .py is compared, and that is the limit of this check: a change under
# csrc/ or CMakeLists.txt lands in a compiled .so that no amount of file
# comparison here will notice. If the checkout moved C++ or build files, the
# wheel needs rebuilding and only you know that.
if [ -d "$VLLM_DIR/.git" ] && [ -d "$VLLM_SITE/vllm" ]; then
    vllm_head=$(git -C "$VLLM_DIR" rev-parse HEAD 2>/dev/null)
    built=$(ls -d "$VLLM_SITE"/vllm-*.dist-info 2>/dev/null | head -1)
    built=${built##*/vllm-}; built=${built%.dist-info}
    say
    say "--- vllm source vs runtime ---"
    say "  runtime:  $VLLM_SITE/vllm  (${built:-version unknown})"
    say "  source:   $VLLM_DIR/vllm   ($BRANCH @ ${vllm_head:0:9})"

    # `Files a differ`: present in both, different content -- the case that
    # changes behaviour. `Only in <tree>`: a module the install never got.
    vllm_cmp=$(diff -rq --exclude=__pycache__ "$VLLM_DIR/vllm" "$VLLM_SITE/vllm" 2>/dev/null)
    vllm_differ=$(printf '%s\n' "$vllm_cmp" | grep -c '^Files .*\.py differ')
    vllm_only=$(printf '%s\n' "$vllm_cmp" | grep -c "^Only in $VLLM_DIR/vllm.*\.py")
    if [ "$vllm_differ" = "0" ] && [ "$vllm_only" = "0" ]; then
        say "  python sources are identical. The engine imports this code."
    else
        say "  $vllm_differ .py differ, $vllm_only .py exist only in the tree:"
        printf '%s\n' "$vllm_cmp" | grep '\.py' | sed 's/^/    /' | head -15
        say "  site-packages/vllm is a plain copy, not a link -- the engine keeps"
        say "  importing ITS copy. Reinstall, or edit site-packages directly, but"
        say "  do not assume this checkout changed what runs."
        NOTES+=("vllm: $vllm_differ .py differ between $VLLM_DIR and the installed copy")
    fi
fi

# ----------------------------------------------------------------- jit wipe
# Opt-in, and only the .so -- the same thing AITER_REBUILD=2 does at import
# time. aiter/jit/build is left alone: it is the ~5 GB of object files that
# make the rebuild minutes instead of tens of minutes, and it is keyed per
# module, so a stale entry there is re-linked rather than re-trusted.
if [ "$CLEAN_JIT" = "1" ] && [ -d "$AITER_DIR/aiter/jit" ]; then
    say
    say "--- CLEAN_JIT=1: removing prebuilt aiter/jit/*.so ---"
    N=$(find "$AITER_DIR/aiter/jit" -maxdepth 1 -name '*.so' -print -delete 2>/dev/null | wc -l)
    say "  removed $N .so (aiter/jit/build kept -- the rebuild links from it)"
    say "  the first call into each kernel now pays a JIT build."
    NOTES+=("CLEAN_JIT removed $N prebuilt .so; first run rebuilds")
fi

# ----------------------------------------------------------------- summary
# The point of the script. arm.json records these same shas at run time, so
# this is what makes the next results directory attributable.
say
echo "==============================================================="
printf " %-8s %-18s %-22s %-10s %s\n" repo state branch head tree
for row in "${SUMMARY[@]}"; do
    IFS='|' read -r a b c d e <<<"$row"
    printf " %-8s %-18s %-22s %-10s %s\n" "$a" "$b" "$c" "$d" "$e"
done
if [ ${#NOTES[@]} -gt 0 ]; then
    say
    for n in "${NOTES[@]}"; do say " note: $n"; done
fi
say
if [ "$RC" -eq 0 ]; then
    say " All three trees are on $BRANCH at the fork's tip. aiter and ATOM are"
    say " pip -e installed, so those are live now -- no reinstall. vllm is NOT:"
    say " the engine imports $VLLM_SITE/vllm regardless."
    say " Next: ./check_image_deps.sh in a fresh container, then a serve script."
else
    say " RESTORE INCOMPLETE -- see above. Do not measure against this set: the"
    say " repos are not guaranteed to be on the same branch."
fi
echo "==============================================================="

exit "$RC"

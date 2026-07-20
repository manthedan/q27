# q27 bench/report shared helpers. Sourced by q27-bench and q27-report.
# macOS-only (Apple silicon). No GPU work happens in this file; it captures
# machine state and locates binaries/models. Keep it dependency-light: the
# whole point is that a friend with a stock Mac can run it.
#
# Conventions:
#   Q27_HOME     model cache root (default ~/.q27/models)
#   Q27_BIN_DIR  where q27-metal / q27-metal-server / benches live
#                (default: same dir as this lib's parent's bin, or PATH)
# Every captured fact is emitted as a `key: value` line so a report bundle
# is trivially greppable and diffable across machines.

# ---------- locations ----------
q27_home() { printf '%s' "${Q27_HOME:-$HOME/.q27/models}"; }

# Find a binary: explicit Q27_BIN_DIR, then next to this script's install
# (…/bin), then PATH.
q27_bin() {
    local name="$1"
    if [ -n "${Q27_BIN_DIR:-}" ] && [ -x "$Q27_BIN_DIR/$name" ]; then
        printf '%s' "$Q27_BIN_DIR/$name"; return 0
    fi
    # lib/ -> ../bin, then a source checkout's build/ dir (dev convenience;
    # absent in an installed formula), then PATH.
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -x "$here/../bin/$name" ]; then printf '%s' "$here/../bin/$name"; return 0; fi
    if [ -x "$here/$name" ]; then printf '%s' "$here/$name"; return 0; fi
    if [ -x "$here/../../build/$name" ]; then printf '%s' "$here/../../build/$name"; return 0; fi
    command -v "$name" 2>/dev/null && return 0
    return 1
}

# ---------- machine header ----------
# Emitted at the top of every report. Root-free.
q27_machine_header() {
    echo "# q27 machine header"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname: $(uname -srm)"
    echo "macos: $(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
    echo "arch: $(uname -m)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "hw_model: $(sysctl -n hw.model 2>/dev/null || echo unknown)"
    echo "ram_gb: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
    echo "ram_bytes: $(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    echo "perflevels: $(sysctl -n hw.nperflevels 2>/dev/null || echo '?')"
    echo "physicalcpu: $(sysctl -n hw.physicalcpu 2>/dev/null || echo '?')"
    local gpucores
    gpucores="$(/usr/sbin/ioreg -r -c AGXAccelerator 2>/dev/null | sed -n 's/.*"gpu-core-count" = \([0-9]*\).*/\1/p' | head -1)"
    echo "gpu_cores: ${gpucores:-?}"
    # q27 binary + shader versions, if resolvable
    local cli; cli="$(q27_bin q27-metal 2>/dev/null || true)"
    echo "q27_metal_bin: ${cli:-MISSING}"
    echo "q27_metal_md5: $( [ -n "$cli" ] && md5 -q "$cli" 2>/dev/null || echo MISSING )"
    echo "git_head: ${Q27_GIT_HEAD:-unknown}"
}

# ---------- thermal / power anchor ----------
# Root-free thermal snapshot (pmset). Optional powermetrics sampler is the
# caller's concern; here we only anchor start/end so a contaminated wall is
# attributable after the fact (the gate-3 / 12.66->10.57 lesson).
q27_thermal_snapshot() {
    local tag="$1"
    if command -v pmset >/dev/null 2>&1; then
        # Flatten to one line; strip the "Note: " prefixes.
        pmset -g therm 2>/dev/null | tr '\n' ' ' | sed 's/Note: //g' | sed 's/  */ /g' \
            | sed "s/^/thermal_$tag: /" || echo "thermal_$tag: unavailable"
    else
        echo "thermal_$tag: unavailable (no pmset)"
    fi
}

# ---------- registry access ----------
# Read one field for a pack name from models.tsv. $1=name $2=column-number
q27_registry_field() {
    local name="$1" col="$2" reg
    reg="$(q27_registry_path)" || return 1
    awk -F'\t' -v n="$name" -v c="$col" \
        '$1 !~ /^#/ && $1 != "name" && $1 == n { print $c; found=1 } END { if(!found) exit 1 }' "$reg"
}
# Locate the python tools used by repack-tier fetches. In a source checkout
# they live in tools/; the formula installs them to pkgshare/q27-tools.
q27_tool_py() {
    local name="$1" here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for cand in "$here/../../tools/$name" "$here/../share/q27-tools/$name" \
                "$here/../q27-tools/$name"; do
        [ -f "$cand" ] && { printf '%s' "$cand"; return 0; }
    done
    return 1
}
q27_bin_repack() { q27_tool_py repack.py; }
q27_bin_export_tok() { q27_tool_py export_tokenizer.py; }

q27_registry_path() {
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for cand in "$here/../models.tsv" "$here/models.tsv" \
                "${Q27_REGISTRY:-}" "$here/../../packaging/models.tsv"; do
        [ -n "$cand" ] && [ -f "$cand" ] && { printf '%s' "$cand"; return 0; }
    done
    return 1
}
q27_registry_names() {
    local reg; reg="$(q27_registry_path)" || return 1
    awk -F'\t' '$1 !~ /^#/ && $1 != "name" && NF > 3 { print $1 }' "$reg"
}

# Resolve a pack name to its artifact path under Q27_HOME. Prints nothing
# (exit 1) if not installed.
q27_artifact_path() {
    local name="$1" f
    f="$(q27_registry_field "$name" 4)" || return 1
    local p="$(q27_home)/$name/$f"
    [ -f "$p" ] && { printf '%s' "$p"; return 0; }
    return 1
}
q27_tokenizer_path() {
    local name="$1" dir tokkind base
    dir="$(q27_home)/$name"
    tokkind="$(q27_registry_field "$name" 10)" || return 1
    # Direct tiers ship qwen36-27b-mtp.tok; repack tiers export <display>.tok.
    # Accept any single .tok present in the dir.
    for t in "$dir"/*.tok; do [ -f "$t" ] && { printf '%s' "$t"; return 0; }; done
    return 1
}

#!/usr/bin/env bash
#
# test_1 -- Celia et al. (1990) Fig. 6(b), 1 m column.
#
# Runs the same three schemes test_richards.py drives through Context, one
# directory each, then draws the figure.
#
#   ./run_test_1.sh                       # all three schemes + figure
#   SCHEMES="FCT" ./run_test_1.sh         # one scheme
#   PLOT_ONLY=1 ./run_test_1.sh           # re-draw from existing archives
#
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P_FILE="re_vgm_sand_10m_1d_p.py"
N_FILE="re_vgm_sand_10m_1d_c0p1_n.py"

SCHEMES="${SCHEMES:-stab_0 stab_2 FCT}"
PLOT_ONLY="${PLOT_ONLY:-0}"
PLOT_PY="${PLOT_PY:-python}"

ctx_for() {
    case "$1" in
        stab_0) echo "STABILIZATION_TYPE=0 FCT=False" ;;   # Galerkin
        stab_2) echo "STABILIZATION_TYPE=2 FCT=False" ;;   # entropy viscosity
        FCT)    echo "STABILIZATION_TYPE=2 FCT=True"  ;;   # entropy viscosity + limiter
        *) echo "unknown scheme '$1'" >&2; return 1 ;;
    esac
}

for scheme in $SCHEMES; do
    ctx="$(ctx_for "$scheme")" || exit 1
    [ "$PLOT_ONLY" = "1" ] && continue

    echo "=== test_1 / $scheme : -C \"$ctx\" ==="
    rundir="$HERE/$scheme"
    mkdir -p "$rundir"
    cp "$HERE/$P_FILE" "$HERE/$N_FILE" "$rundir/"
    (
        cd "$rundir" || exit 1
        parun -p "$P_FILE" "$N_FILE" -l 5 -v -C "$ctx" 2>&1 | tee run.log
        exit "${PIPESTATUS[0]}"
    ) || echo "!!! test_1/$scheme FAILED -- see $rundir/run.log" >&2
done

echo "=== figure ==="
SCHEMES="$SCHEMES" $PLOT_PY "$HERE/plot_test_1.py"

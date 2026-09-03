#!/usr/bin/env bash
#
# Tracy transient Richards spatial-convergence sweep.
#
#   ./run_tracy_convergence.sh                    # all 3 schemes, ref_3..ref_6
#   SCHEMES="FCT" ./run_tracy_convergence.sh      # one scheme
#   REFS="ref_3 ref_4" ./run_tracy_convergence.sh # a subset of refinements
#   ANALYZE_ONLY=1 ./run_tracy_convergence.sh     # skip the runs, just re-tabulate
#
# Layout produced (relative to this directory):
#   <scheme>/ref_N/{re_vgm_*.py, *.h5, mpi_run.log}
#   <scheme>/Tracy_L2_convergence.txt
#
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P_FILE="re_vgm_sand_10x10m_2d_p.py"
N_FILE="re_vgm_sand_10x10m_2d_c0p1_n.py"

SCHEMES="${SCHEMES:-Stab_0 Stab_2 FCT}"
REFS="${REFS:-ref_3 ref_4 ref_5 ref_6}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
PETSC_OPTS="${PETSC_OPTS:--ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist}"
# the error analysis needs quadpy, which lives in its own env
ANALYZE_PY="${ANALYZE_PY:-conda run -n quadpy_env python}"

# scheme -> context options
ctx_for() {
    case "$1" in
        Stab_0) echo "STABILIZATION_TYPE=0 FCT=False" ;;   # Galerkin
        Stab_2) echo "STABILIZATION_TYPE=2 FCT=False" ;;   # entropy viscosity
        FCT)    echo "STABILIZATION_TYPE=2 FCT=True"  ;;   # EV + FCT limiter
        *) echo "unknown scheme '$1'" >&2; return 1 ;;
    esac
}

# refinement -> nnx, MPI ranks
nnx_for() {
    case "$1" in
        ref_2) echo 41  ;;
        ref_3) echo 81  ;;
        ref_4) echo 161 ;;
        ref_5) echo 321 ;;
        ref_6) echo 641 ;;
        *) echo "unknown refinement '$1'" >&2; return 1 ;;
    esac
}

np_for() {
    case "$1" in
        ref_2) echo 4  ;;
        ref_3) echo 8  ;;
        ref_4) echo 16 ;;
        ref_5) echo 64 ;;
        ref_6) echo 90 ;;
        *) echo "unknown refinement '$1'" >&2; return 1 ;;
    esac
}

for scheme in $SCHEMES; do
    ctx="$(ctx_for "$scheme")" || exit 1
    for ref in $REFS; do
        nnx="$(nnx_for "$ref")" || exit 1
        np="$(np_for "$ref")"   || exit 1
        rundir="$HERE/$scheme/$ref"

        if [ "$ANALYZE_ONLY" = "1" ]; then continue; fi

        echo "=== $scheme / $ref : nnx=$nnx, -n $np, -C \"$ctx nnx=$nnx\" ==="
        mkdir -p "$rundir"
        cp "$HERE/$P_FILE" "$HERE/$N_FILE" "$rundir/"
        (
            cd "$rundir" || exit 1
            start=$SECONDS
            mpiexec -n "$np" parun -p "$P_FILE" "$N_FILE" \
                -l 5 -v \
                -C "$ctx nnx=$nnx" \
                -P "$PETSC_OPTS" 2>&1 | tee mpi_run.log
            status=${PIPESTATUS[0]}
            echo "wall_seconds $(( SECONDS - start ))" >> mpi_run.log
            exit "$status"
        )
        if [ $? -ne 0 ]; then
            echo "!!! $scheme/$ref FAILED — see $rundir/mpi_run.log" >&2
        fi
    done
done

# ---------------------------------------------------------------- errors
first_ref="${REFS%% *}"
last_ref="${REFS##* }"
for scheme in $SCHEMES; do
    [ -d "$HERE/$scheme" ] || continue
    echo "=== L2 convergence: $scheme ==="
    (
        cd "$HERE/$scheme" || exit 1
        REF_MIN="$first_ref" REF_MAX="$last_ref" \
            $ANALYZE_PY "$HERE/analyze_convergence_L2.py"
    )
done

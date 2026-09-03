#!/usr/bin/env bash
#
# Raingarden -- 3 x 5 m two-layer rain garden, fine over coarse at z = 3 m.
#
# Runs the three schemes through Context, one directory each, then draws the two
# figures:
#   fct_nnx41_pressure_head_panels.png   infiltration at three times (FCT run)
#   hydrus_vs_schemes_contours.png       contour lines vs HYDRUS-2D/3D
#
#   ./run_raingarden.sh                       # all three schemes + both figures
#   SCHEMES="FCT" ./run_raingarden.sh         # one scheme
#   NP=2 ./run_raingarden.sh                  # MPI ranks per run
#   PLOT_ONLY=1 ./run_raingarden.sh           # re-draw from existing archives
#   PETSC_OPTS="..." ./run_raingarden.sh      # override the linear solver
#
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P_FILE="Raingarden_p.py"
N_FILE="Raingarden_n.py"
D_FILE="domain_rg.py"

SCHEMES="${SCHEMES:-stab_0 stab_2 FCT}"
NP="${NP:-2}"
PLOT_ONLY="${PLOT_ONLY:-0}"
PLOT_PY="${PLOT_PY:-python}"

# Direct sparse factorization, handed to parun as one string via -P.  Without
# these, PETSc falls back to its default GMRES/block-Jacobi, and because
# Raingarden_n.py leaves l_atol_res at the default_n.py value of 1.0 the KSP
# converges on atol with its=0 and hands back a zero correction -- Newton then
# sits at a constant residual until maxIts.
PETSC_OPTS="${PETSC_OPTS:--ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist}"

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

    echo "=== raingarden / $scheme : -n $NP, -C \"$ctx\" ==="
    rundir="$HERE/$scheme"
    mkdir -p "$rundir"
    cp "$HERE/$P_FILE" "$HERE/$N_FILE" "$HERE/$D_FILE" "$rundir/"
    (
        cd "$rundir" || exit 1
        mpiexec -n "$NP" parun -p "$P_FILE" "$N_FILE" -l 5 -v -C "$ctx" -P "$PETSC_OPTS" 2>&1 \
            | tee mpi_run.log
        exit "${PIPESTATUS[0]}"
    ) || echo "!!! raingarden/$scheme FAILED -- see $rundir/mpi_run.log" >&2
done

echo "=== figure 1: infiltration panels (FCT) ==="
SCHEME=FCT $PLOT_PY "$HERE/plot_infiltration_panels.py"

echo "=== figure 2: contour lines vs HYDRUS ==="
$PLOT_PY "$HERE/compare_hydrus_vs_schemes.py"

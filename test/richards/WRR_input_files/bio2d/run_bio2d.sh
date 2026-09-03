#!/usr/bin/env bash
#
# bio2d -- 3 x 5 m two-dimensional bioswale, trapezoidal swale over a three-layer
# profile with a rectangular drain.
#
# Runs the three schemes through Context, one directory each, then draws the
# figures and prints the two tables:
#   bio2d_psi_<a-f>_t<t>.png        pressure head at six times (FCT run)
#   hydrus_vs_schemes_contours.png  contour lines vs the HYDRUS reference
#   deviation_table.py              L2 deviation from HYDRUS, per scheme
#   table_from_log.py               solver cost, per scheme
#
#   ./run_bio2d.sh                        # all three schemes + figures + tables
#   SCHEMES="FCT" ./run_bio2d.sh          # one scheme
#   HE=0.15 ./run_bio2d.sh                # finer mesh (default he = 0.3 m)
#   MESH=hydrus ./run_bio2d.sh            # solve on the HYDRUS mesh -> hydrus_mesh/
#   NP=2 ./run_bio2d.sh                   # MPI ranks per run
#   PLOT_ONLY=1 ./run_bio2d.sh            # re-draw/re-tabulate from what is on disk
#   PETSC_OPTS="..." ./run_bio2d.sh       # override the linear solver
#
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P_FILE="Raingarden_p.py"
N_FILE="Raingarden_n.py"
D_FILE="domain_rg.py"

SCHEMES="${SCHEMES:-Stab_0 Stab_2 FCT}"
HE="${HE:-0.3}"
NP="${NP:-2}"
PLOT_ONLY="${PLOT_ONLY:-0}"
PLOT_PY="${PLOT_PY:-python}"

# MESH=triangle (default) meshes the PSLG at he.  MESH=hydrus solves on the
# HYDRUS triangulation itself (bio2d_H.node/.ele/.edge) so the two codes share a
# node set and the deviation table is node-to-node exact; he is then ignored.
# Those runs go in hydrus_mesh/ so both families can sit on disk at once, and the
# analysis scripts follow through BIO2D_RUNS.
MESH="${MESH:-triangle}"
if [ "$MESH" = "hydrus" ]; then
    RUNS_DIR="${RUNS_DIR:-$HERE/hydrus_mesh}"
    CTX_MESH="mesh='hydrus'"
else
    RUNS_DIR="${RUNS_DIR:-$HERE}"
    CTX_MESH="he=$HE"
fi
mkdir -p "$RUNS_DIR"
export BIO2D_RUNS="$RUNS_DIR"

# Direct sparse factorization, handed to parun as one string via -P.  Without
# these, PETSc falls back to its default GMRES/block-Jacobi, and because
# Raingarden_n.py leaves l_atol_res at the default_n.py value of 1.0 the KSP
# converges on atol with its=0 and hands back a zero correction -- Newton then
# sits at a constant residual until maxIts.
PETSC_OPTS="${PETSC_OPTS:--ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist}"

ctx_for() {
    case "$1" in
        Stab_0) echo "STABILIZATION_TYPE=0 FCT=False" ;;   # Galerkin
        Stab_2) echo "STABILIZATION_TYPE=2 FCT=False" ;;   # entropy viscosity
        FCT)    echo "STABILIZATION_TYPE=2 FCT=True"  ;;   # entropy viscosity + limiter
        *) echo "unknown scheme '$1'" >&2; return 1 ;;
    esac
}

for scheme in $SCHEMES; do
    ctx="$(ctx_for "$scheme")" || exit 1
    [ "$PLOT_ONLY" = "1" ] && continue

    echo "=== bio2d / $scheme : -n $NP, -C \"$ctx $CTX_MESH\" ==="
    rundir="$RUNS_DIR/$scheme"
    mkdir -p "$rundir"
    cp "$HERE/$P_FILE" "$HERE/$N_FILE" "$HERE/$D_FILE" "$rundir/"
    (
        cd "$rundir" || exit 1
        # -p is proteus's PROFILE flag, and it is not optional here: table_from_log.py
        # reads the assembly time and the CPU time out of the profile block that it
        # appends to Raingarden_p.log.  Every scheme is run with it so the cost table
        # compares like with like.
        mpiexec -n "$NP" parun -p "$P_FILE" "$N_FILE" -l 5 -v -C "$ctx $CTX_MESH" -P "$PETSC_OPTS" 2>&1 \
            | tee mpi_run.log
        exit "${PIPESTATUS[0]}"
    ) || echo "!!! bio2d/$scheme FAILED -- see $rundir/mpi_run.log" >&2
done

echo "=== figure 1: pressure-head panels (FCT) ==="
SCHEME=FCT $PLOT_PY "$HERE/plot_psi_six_times.py"

echo "=== figure 2: contour lines vs HYDRUS ==="
$PLOT_PY "$HERE/compare_hydrus_contours.py"

echo "=== table 1: L2 deviation from HYDRUS ==="
$PLOT_PY "$HERE/deviation_table.py" --latex

echo "=== table 2: solver cost ==="
$PLOT_PY "$HERE/table_from_log.py"

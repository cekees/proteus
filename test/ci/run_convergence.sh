#!/usr/bin/env bash
#
# Generalized IFEM/SCIFEM convergence driver for the ladr_ss_2d test family.
#
# Replaces the per-test ex*.sh scripts, which were near-identical copies of the
# same six blocks differing only in the `test=` value.  Each block is
#
#   parun ladr_ss_2d_p.py ladr_ss_2d_c0p<order>_n.py -l 5 -v \
#         -C "test=<T> <mesh options> refinement=<r>"
#
# swept over refinement = 1..6, for order = P1 and P2, on three meshes.
# Since ladr_ss_2d_c0p*_n.py sets nnx = 4*2**refinement + 1, h is halved
# exactly once per refinement, so the observed order is log2(e_r / e_{r+1}).
#
# Usage:
#   ./run_convergence.sh <test> [options]
#   ./run_convergence.sh all    [options]
#
# Options:
#   -o, --order <1|2|both>   IFEM order (default: both; prompts if stdin is a tty
#                            and --order was not given, matching the old ex*.sh)
#   -m, --mesh <list>        comma-separated mesh configs (default: us,usr,str)
#                              us   unstructured, default skew  -> ex<T>.out
#                              usr  unstructured, skew=0.0      -> ex<T>ru.out
#                              str  structured                  -> ex<T>r.out
#   -r, --refinements <list> space/comma separated (default: 1 2 3 4 5 6)
#   -s, --scifem <0.0|1.0>   immersedSCIFEM_switch; must be a float -- pass 1.0, not 1
#   -p, --penalty <gamma>    immersedSCIFEM_penalty, likewise a float (e.g. 10.0)
#   -C, --extra "<opts>"     extra -C options appended verbatim
#   -k, --keep               keep the per-refinement logs
#   -h, --help               this message
#
# Examples:
#   ./run_convergence.sh 8                    # test=8.0, P1 and P2, all 3 meshes
#   ./run_convergence.sh 8 -o 1 -m str        # P1 on the structured mesh only
#   ./run_convergence.sh 12 -o 2 -s 1.0 -p 10.0   # SCIFEM on, penalty 10
#   ./run_convergence.sh all -o 1 -m str -r "1 2 3 4"
#
set -uo pipefail

VALID_TESTS="1.0 2.0 2.1 3.0 4.0 4.1 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0"

# print the header comment block, stopping at the first non-comment line
usage() { awk 'NR>=3 && /^#/ { sub(/^# ?/, ""); print; next } NR>=3 { exit }' "$0"; }

ORDER=""
MESHES="us,usr,str"
REFINEMENTS="1 2 3 4 5 6"
SCIFEM=""
PENALTY=""
EXTRA=""
KEEP=0
TEST_ARG=""

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--order)       ORDER="$2"; shift 2 ;;
        -m|--mesh)        MESHES="$2"; shift 2 ;;
        -r|--refinements) REFINEMENTS="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
        -s|--scifem)      SCIFEM="$2"; shift 2 ;;
        -p|--penalty)     PENALTY="$2"; shift 2 ;;
        -C|--extra)       EXTRA="$2"; shift 2 ;;
        -k|--keep)        KEEP=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        -*)               echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
        *)                if [ -n "$TEST_ARG" ]; then
                              echo "only one test may be given (got '$TEST_ARG' and '$1')" >&2
                              exit 1
                          fi
                          TEST_ARG="$1"; shift ;;
    esac
done

if [ -z "$TEST_ARG" ]; then
    echo "error: no test given" >&2
    usage >&2
    exit 1
fi

# "8" -> "8.0", "2.1" -> "2.1"
normalize_test() {
    case "$1" in
        *.*) printf '%s' "$1" ;;
        *)   printf '%s.0' "$1" ;;
    esac
}

if [ "$TEST_ARG" = "all" ]; then
    TEST_LIST="$VALID_TESTS"
else
    TEST_LIST="$(normalize_test "$TEST_ARG")"
    case " $VALID_TESTS " in
        *" $TEST_LIST "*) ;;
        *) echo "error: unknown test '$TEST_ARG'; ladr_ss_2d_p.py defines: $VALID_TESTS" >&2
           exit 1 ;;
    esac
fi

# Match the old ex*.sh interactive prompt when no order was given.
if [ -z "$ORDER" ]; then
    if [ -t 0 ]; then
        echo -n "Enter IFEM order to test (1, 2, or both): "
        read -r ORDER
        [ -z "$ORDER" ] && ORDER="both"
    else
        ORDER="both"
    fi
fi
case "$ORDER" in
    1)    ORDER_LIST="1" ;;
    2)    ORDER_LIST="2" ;;
    both) ORDER_LIST="1 2" ;;
    *)    echo "Invalid input. Please enter 1, 2, or both." >&2; exit 1 ;;
esac

SCIFEM_OPTS=""
[ -n "$SCIFEM"  ] && SCIFEM_OPTS="$SCIFEM_OPTS immersedSCIFEM_switch=$SCIFEM"
[ -n "$PENALTY" ] && SCIFEM_OPTS="$SCIFEM_OPTS immersedSCIFEM_penalty=$PENALTY"
[ -n "$EXTRA"   ] && SCIFEM_OPTS="$SCIFEM_OPTS $EXTRA"

mesh_opts()  { case "$1" in
                   us)  echo "unstructured=True" ;;
                   usr) echo "unstructured=True skew=0.0" ;;
                   str) echo "unstructured=False" ;;
               esac; }
mesh_label() { case "$1" in
                   us)  echo "unstructured (default skew)" ;;
                   usr) echo "unstructured (skew=0.0)" ;;
                   str) echo "structured" ;;
               esac; }
mesh_suffix(){ case "$1" in us) echo "" ;; usr) echo "ru" ;; str) echo "r" ;; esac; }

# `[  0.7602] L2 error = [0.04873722]` -> `0.04873722`
extract() { grep "$2 error = " "$1" | tail -1 | sed -E 's/.*= *\[ *([0-9.eE+-]+).*/\1/'; }

print_table() {
    awk -v hdr="$1" -v refs="$2" -v l2s="$3" -v lis="$4" '
    function rate(prev, cur, i,   p) {
        p = prev + 0
        if (i <= 1 || cur <= 0 || p <= 0)   return sprintf("%7s", "-")
        if (cur < 1e-12 && p < 1e-12)       return sprintf("%7s", "exact")
        return sprintf("%7.2f", log(p / cur) / log(2.0))
    }
    BEGIN {
        n = split(refs, R, " "); split(l2s, A, " "); split(lis, B, " ")
        printf "\n%s\n", hdr
        printf "  %-4s %-6s %14s %7s %16s %7s\n", "ref", "nnx", "L2 error", "rate", "Linfty error", "rate"
        printf "  %s\n", "-------------------------------------------------------------------"
        for (i = 1; i <= n; i++) {
            nnx = 4 * 2^R[i] + 1
            a = A[i] + 0; b = B[i] + 0
            as = (A[i] == "" || A[i] == "FAIL") ? "   FAILED" : sprintf("%14.6e", a)
            bs = (B[i] == "" || B[i] == "FAIL") ? "   FAILED" : sprintf("%16.6e", b)
            # Below ~1e-12 the FE space reproduces the solution exactly and what is
            # left is roundoff, so a "rate" there is noise, not an order.
            r1 = rate(A[i-1], a, i); r2 = rate(B[i-1], b, i)
            printf "  %-4s %-6d %s %s %s %s\n", R[i], nnx, as, r1, bs, r2
        }
    }'
}

LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/run_convergence.XXXXXX")"
trap '[ "$KEEP" -eq 0 ] && rm -rf "$LOGDIR"' EXIT
STATUS=0

for TEST in $TEST_LIST; do
    for ORD in $ORDER_LIST; do
        NFILE="ladr_ss_2d_c0p${ORD}_n.py"
        SUMMARY="p${ORD}.out"
        echo "Test=${TEST} IFEM order ${ORD}" > "$SUMMARY"

        for M in $(echo "$MESHES" | tr ',' ' '); do
            case "$M" in us|usr|str) ;; *) echo "unknown mesh '$M'" >&2; exit 1 ;; esac
            MOPTS="$(mesh_opts "$M")"
            OUTFILE="ex${TEST}$(mesh_suffix "$M").out"
            HDR="=== test=${TEST}  P${ORD}  $(mesh_label "$M") ==="
            [ -n "$SCIFEM_OPTS" ] && HDR="$HDR  [${SCIFEM_OPTS# }]"

            echo "--------------------------------------------------------------"
            echo "$HDR"
            : > "$OUTFILE"
            L2S=""; LIS=""
            for R in $REFINEMENTS; do
                LOG="$LOGDIR/t${TEST}_p${ORD}_${M}_r${R}.log"
                printf '  refinement=%s ... ' "$R"
                if parun ladr_ss_2d_p.py "$NFILE" -l 5 -v \
                        -C "test=${TEST} ${MOPTS} refinement=${R}${SCIFEM_OPTS}" \
                        > "$LOG" 2>&1; then
                    L2="$(extract "$LOG" L2)"; LI="$(extract "$LOG" Linfty)"
                    [ -z "$L2" ] && L2="FAIL"
                    [ -z "$LI" ] && LI="FAIL"
                    echo "L2=$L2"
                else
                    L2="FAIL"; LI="FAIL"; STATUS=1
                    echo "parun FAILED (see $LOG)"
                    KEEP=1
                fi
                cat "$LOG" >> "$OUTFILE"
                L2S="$L2S $L2"; LIS="$LIS $LI"
            done

            print_table "$HDR" "$REFINEMENTS" "$L2S" "$LIS" | tee -a "$SUMMARY"
            grep "L2 error = "     "$OUTFILE" >> "$SUMMARY"
            grep "Linfty error = " "$OUTFILE" >> "$SUMMARY"
        done
        echo
        echo "summary written to $SUMMARY"
    done
done

[ "$KEEP" -eq 1 ] && echo "logs kept in $LOGDIR"
exit $STATUS

#!/bin/bash
# usage: [HS_LAG=0] [HS_RD=0] HS_TAG=_x run_matrix.sh PYTHON ROOT TI TORDER N
# Runs full, full (again, for the determinism floor), and stop@N + hotstart,
# then compares per field.
PY=$1; ROOT=$2; TI=$3; TO=$4; N=${5:-40}
D=$(cd "$(dirname "$0")" && pwd); K=${TI}${TO}; T=${HS_TAG:-}; B=hs_${K}${T}
mkdir -p "$ROOT"
$PY $D/run_case.py $ROOT/${K}${T}_full  $TI $TO full     > $ROOT/${K}${T}_full.log 2>&1
$PY $D/run_case.py $ROOT/${K}${T}_full2 $TI $TO full     > $ROOT/${K}${T}_full2.log 2>&1
$PY $D/run_case.py $ROOT/${K}${T}_hot   $TI $TO stop $N  > $ROOT/${K}${T}_stop.log 2>&1
$PY $D/run_case.py $ROOT/${K}${T}_hot   $TI $TO hot  $N  > $ROOT/${K}${T}_hot.log 2>&1
echo "== $K$T determinism"; $PY $D/compare.py $ROOT/${K}${T}_full/$B.h5 $ROOT/${K}${T}_full2/$B.h5 $K$T $((N+1)) | tail -1
echo "== $K$T hotstart"; $PY $D/compare.py $ROOT/${K}${T}_full/$B.h5 $ROOT/${K}${T}_hot/$B.h5 $K$T $N | grep -E "^(41|FINAL)"

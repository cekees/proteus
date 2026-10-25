#!/bin/bash
# usage: oracle.sh PYTHON ROOT TI TORDER N GROUPS...
# Hotstart, then overwrite the named groups of state with the uninterrupted
# run's values at step N. A group whose restoration removes the error is
# state the archive must carry. "none" is the plain hotstart.
PY=$1; ROOT=$2; TI=$3; TO=$4; N=$5; shift 5
D=$(cd "$(dirname "$0")" && pwd); K=${TI}${TO}; B=hs_${K}
if [ ! -f $ROOT/${K}_full/$B.h5 ]; then $PY $D/run_case.py $ROOT/${K}_full $TI $TO full > $ROOT/${K}_full.log 2>&1; fi
rm -rf $ROOT/or_stop; HS_SNAP=$ROOT/${K}_true.pkl $PY $D/run_case.py $ROOT/or_stop $TI $TO stop $N > $ROOT/or_stop.log 2>&1
for g in "$@"; do
  rm -rf $ROOT/or_$g; cp -r $ROOT/or_stop $ROOT/or_$g
  if [ "$g" = none ]; then
    $PY $D/run_case.py $ROOT/or_$g $TI $TO hot $N > $ROOT/or_$g.log 2>&1
  else
    HS_ORACLE=$g HS_TRUE=$ROOT/${K}_true.pkl $PY $D/run_case.py $ROOT/or_$g $TI $TO hot $N > $ROOT/or_$g.log 2>&1
  fi
  echo "-- oracle: $g  $(grep ORACLE $ROOT/or_$g.log)"
  $PY $D/compare.py $ROOT/${K}_full/$B.h5 $ROOT/or_$g/$B.h5 $g $N | grep -E "^($((N+1))|80) " | grep -E " (u|phid|vof) "
done

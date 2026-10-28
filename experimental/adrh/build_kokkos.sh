#!/bin/sh
# Build the Kokkos benchmark (and, with a Python that has pybind11, the adrh_kokkos module) for a CPU Kokkos.
#   sh build_kokkos.sh                inside a conda environment with `kokkos` installed ($CONDA_PREFIX)
#   KOKKOS=/path/to/install sh build_kokkos.sh      a Kokkos installed elsewhere (e.g. built from source)
# A device build (CUDA, HIP) needs the compiler Kokkos was configured with (nvcc_wrapper, hipcc) and
# its flags: see ADRH_ENGINE_DESIGN.md section 19 for the CUDA recipe used on ceds-model.
set -e
cd "$(dirname "$0")"
K=${KOKKOS:-$CONDA_PREFIX}
CXX=${CXX:-c++}
OMP=""; LIBS="-lkokkoscore -lkokkossimd -lhwloc"
if [ "$(uname)" = Darwin ]; then OMP="-Xclang -fopenmp"; LIBS="$LIBS -lomp"; else OMP="-fopenmp"; fi
FLAGS="-O3 -std=c++20 $OMP -I$K/include -L$K/lib -Wl,-rpath,$K/lib"
$CXX $FLAGS bench_kokkos.cpp -o bench_kokkos_cpu $LIBS
echo "built bench_kokkos_cpu"
if python -c "import pybind11" 2>/dev/null; then
  EXT=$(python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
  $CXX $FLAGS -shared -fPIC $([ "$(uname)" = Darwin ] && echo -undefined dynamic_lookup) $(python -m pybind11 --includes) \
    module_kokkos.cpp -o adrh_kokkos$EXT $LIBS
  echo "built adrh_kokkos$EXT"
fi

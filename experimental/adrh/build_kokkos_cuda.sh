#!/bin/sh
# Build Kokkos from source with the OpenMP host backend and CUDA, and build and test the ADRH modules against it.
# conda-forge's CUDA Kokkos packages cannot be solved by conda (they need the __cuda_arch virtual package, which only
# mamba provides) and are not built for sm86/sm89; a source build takes a couple of minutes and any architecture.
#
#   sh build_kokkos_cuda.sh ARCH            e.g. TURING75 (T4), AMPERE80, ADA89, HOPPER90
#
# Run it in a conda environment with cmake, ninja, gxx_linux-64, python, pybind11, numpy, scipy, cuda-nvcc, cuda-cudart-dev
# and cuda-version (the CI job in .github/workflows/adrh.yml has the list). KOKKOS_SRC, if set, is an unpacked Kokkos
# source tree; otherwise 5.2.2 is downloaded. WORK (default ./gpu-build) holds the Kokkos install and the build.
set -e
ARCH=${1:?usage: sh build_kokkos_cuda.sh ARCH (TURING75, AMPERE80, ADA89, HOPPER90, ...)}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$PWD/gpu-build}
VERSION=5.2.2
SHA256=491fe02cb127559dd3c996f6693686f1eab00a53fd5c184e18e663896884175e
mkdir -p "$WORK"
if [ -z "$KOKKOS_SRC" ]; then
  if [ ! -d "$WORK/kokkos-$VERSION" ]; then
    curl -fsSL -o "$WORK/kokkos.tar.gz" "https://github.com/kokkos/kokkos/archive/refs/tags/$VERSION.tar.gz"
    echo "$SHA256  $WORK/kokkos.tar.gz" | (sha256sum -c - 2>/dev/null || shasum -a 256 -c -)
    tar -xzf "$WORK/kokkos.tar.gz" -C "$WORK"
  fi
  KOKKOS_SRC=$WORK/kokkos-$VERSION
fi
# nvcc needs a host compiler its CUDA release supports and that matches the libstdc++ the Python extensions use:
# the one conda installs, not the system's
HOSTCXX=$(command -v x86_64-conda-linux-gnu-g++ || command -v g++)
export NVCC_WRAPPER_DEFAULT_COMPILER=$HOSTCXX
CUDA_ROOT=${CUDAToolkit_ROOT:-$CONDA_PREFIX/targets/x86_64-linux}
cmake -S "$KOKKOS_SRC" -B "$WORK/kokkos-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="$WORK/kokkos-install" -DCMAKE_CXX_COMPILER="$KOKKOS_SRC/bin/nvcc_wrapper" -DCMAKE_CXX_STANDARD=20 \
  -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_$ARCH=ON \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_ROOT"
cmake --build "$WORK/kokkos-build"
cmake --install "$WORK/kokkos-build"

cmake -S "$HERE" -B "$WORK/adrh-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python)" \
  -DCMAKE_PREFIX_PATH="$WORK/kokkos-install" -DADRH_EXPECT_SPACE=Cuda -DCMAKE_CXX_COMPILER="$WORK/kokkos-install/bin/nvcc_wrapper"
cmake --build "$WORK/adrh-build"
echo "built; run: ctest --test-dir $WORK/adrh-build --output-on-failure"

# ADRH engine prototype

Experimental, not part of the proteus build. A header-only n-component
advection-diffusion-reaction-Hamiltonian element kernel (P1 simplices, Galerkin) with the
coefficient functor as a template parameter, to settle the interface a ymf code generator has to
target. Design: `ADRH_ENGINE_DESIGN.md` (kept beside the campaign notes).

* `adrh.h` – the `Terms<nc,nS>` interface and `elementKernel` (geometry, u and grad u, coefficient
  evaluation and the weak form, residual and Jacobian, all inside the element loop; nothing stored
  per quadrature point).
* `coefficients.h` – hand-written functors in generated form: `Poisson3D` (test/ci/poisson_3d_p)
  and `Coupled2`, a two-component system with every term nonlinear.
* `module.cpp`, `build.sh` – pybind11 module (`sh build.sh` inside the proteus environment).
* `compare_with_python.py [nn]` – checks the kernel against `OneLevelTransport`'s element residual
  and Jacobian on tets, and the coupled system's Jacobian against finite differences.

Measured (Apple M-series, one core): relative difference to the Python path 7e-16 (residual, on
elements with no exterior face: the Python element residual includes the exterior boundary-flux
terms) and 8e-16 (Jacobian); coupled Jacobian vs central differences 1e-10. Element residual and
Jacobian for 162,000 tets: 0.054 s for the kernel including geometry and coefficient evaluation,
against 0.39 s for the Python path's kernels alone with its quadrature arrays already filled.

## Sparsity masks and the advection-diffusion regression

* `masked.h` – `elementKernelMasked`: the same kernel with the functor's sparsity as `static constexpr` predicates
  (`DenseMasks` says everything is nonzero; a functor redefines what it knows). Component loops are unrolled with
  `constexpr_for` and every disallowed block is dropped with `if constexpr`. `compare_masks.py` checks it against the
  runtime-loop kernel and times both: a five-component chain with tridiagonal reaction is 23x faster, a two-component
  coupled system 2.1x, Poisson 1.3x; with all-true masks the unrolled kernel costs the same as the runtime loops, so the gain
  is all sparsity.
* `ad_regression.py` – steady advection-diffusion with the manufactured sine solution
  (`AnalyticalSolutions.LinearADR_Sine`, c = 0) on regular triangles (`AD_ND=2`) or tets (`AD_ND=3`): the engine, a plain
  Proteus model (`ad_sine_p.py`, `ad_sine_n.py`) and the exact solution. Engine and Proteus nodal solutions agree to solver
  round-off (< 5e-14) in every boundary mode and both dimensions; L2 error rates 1.85-2.00.

## Boundary conditions (`boundary.h`)

`exteriorFaceKernel` adds the exterior-face terms per face (residual and Jacobian, scattered by the driver), for
segments of triangles and triangles of tetrahedra (`Coefficients::nSpace` = 2 or 3). A BC functor says, per component
and boundary quadrature point, which of three kinds applies; each is the engine's version of a Proteus exterior
numerical-flux treatment, and `ad_regression.py` compares against that treatment (`AD_BC=`, in 2D and 3D):

| kind | what it does | Proteus equivalent | `AD_BC` |
|---|---|---|---|
| `Strong` | nothing on the face; the driver replaces the row | `StrongDirichletFactory`, `numericalFluxType=None` | `strong` |
| `Flux` | prescribed diffusive flux, optionally affine in the trace (`gdiff + dgdiff u`); advective flux prescribed where given, else the interior `f(u).n` on outflow and zero on inflow | `StrongDirichletFactory({ci: 'outFlow'})` with `diffusiveFluxBoundaryConditions` / `advectiveFluxBoundaryConditions`; `RobinFactory` for `alpha u + beta q.n = g` | `mixed`, `inflow`, `robin` |
| `WeakDirichlet` | upwinded advective flux against `f(g)`, `-a grad phi . n + (C/h^p)(u - g)`, adjoint `-sigma (u - g) (a^T n).grad w` | `Advection_DiagonalUpwind_Diffusion_{IIPG,NIPG,SIPG}_exterior` (C = 2, 10, 100; sigma = 0, -1, +1) | `iipg`, `nipg`, `sipg` |

The header's comment says where the engine deliberately differs from Proteus (the adjoint's tensor contraction, da/du in
the Jacobian); neither shows with an isotropic constant diffusion tensor.

Design, decisions and numbers: `ADRH_ENGINE_DESIGN.md` (kept beside the campaign notes, sections 11-15).

## Kokkos port (threads and devices)

Decision (cekees, 2026-10-24): the generated kernels get on-node threads and devices in 2.0.0, in Kokkos, with element
coloring for the assembly's scatter (`ADRH_ENGINE_DESIGN.md` section 19, roster C19).

* `portable.h` – `ADRH_FN` is `KOKKOS_INLINE_FUNCTION` when `ADRH_USE_KOKKOS` is defined and `inline` otherwise, so the
  headers are the host prototype's and the device kernel's, one source. Nothing else in the kernel changed.
* `masked.h` – the body of the element loop is now `elementMaskedOne` (one element: coordinates and DOF values in, R and J
  out); `elementKernelMasked` is the loop over it. **It and the compile-time loop helpers are `always_inline`**: left to the
  compiler's heuristics, a helper (`constexpr_for` with its lambdas) stayed a real call whenever the kernel was reached through a member
  function (a Kokkos functor), which pushes every array of the element into memory. The Poisson kernel ran 3.2 times
  slower there (815 against 250 ns per element) than in a plain loop; Kokkos was not the cause. With the fix the host prototype itself is
  faster: masked coupled system 3.4 to 1.3 microseconds per element, masked five-component chain 6.6 to 2.5, the same chain
  with all-true masks 146 to 52 (`compare_masks.py`).
* `kokkos_kernel.h` – `elementKernelKokkos` (one Kokkos iteration per element, R and J stored, the prototype's interface) and
  the fused `assembleColored`: a greedy node-sharing coloring (`colorElements`, 30 colors on the cube's tetrahedra), a block-CSR
  Jacobian pattern (`buildBlockCsr`), and one launch per color in which a thread computes its element and adds it to the global residual and
  Jacobian. Elements of one color share no node, so there are no atomics and the sums happen in the same order every run.
* `bench_kokkos.cpp` – the four functors on a structured tetrahedral mesh (`nn` nodes per edge; 31 is 162,000 tets): host prototype,
  Kokkos element kernel, colored assembly, with the differences and a checksum of the assembled bits.
* `module_kokkos.cpp` – `adrh_kokkos`, the host module's calling convention over the Kokkos kernel (masked kernel only), so the
  existing checks run on it: `ADRH_MODULE=adrh_kokkos python compare_with_python.py | ad_regression.py`; `golden.py` compares
  any two builds (`golden.py against adrh_kokkos`; `golden.py save|compare` brackets a change to the host kernel).
* `build_kokkos.sh` – CPU build in a conda environment with `kokkos`; the CUDA recipe is in the design note.

Results (`reports/adrh_kokkos_2026-10-24/` beside the campaign notes; nn = 31, ns per element, element kernel / colored assembly):

| | Poisson | AD sine | coupled 2 | chain 5 |
|---|---|---|---|---|
| Mac M4 Max, host prototype, 1 thread | 238 | 277 | 1,322 | 2,408 |
| Mac, Kokkos OpenMP, 1 thread | 230 / 251 | 273 / 291 | 1,313 / 1,448 | 2,404 / 2,863 |
| Mac, Kokkos OpenMP, 12 threads | 21 / 35 | 24 / 37 | 118 / 136 | 213 / 272 |
| ceds-model Zen 4, host prototype, 1 thread (g++ -O3) | 578 | 554 | 3,274 | 5,382 |
| ceds-model, Kokkos OpenMP, 96 threads (g++ -O3) | 11 / 16 | 10 / 16 | 52 / 60 | 90 / 105 |
| ceds-model, Kokkos OpenMP, 96 threads (-march=native), nn = 61 | 5.7 / 8.0 | 8.3 / 10.4 | 31 / 34 | 52 / 64 |
| ceds-model, one RTX 6000 Ada, nn = 61 | 8.8 / 10.8 | 8.3 / 10.3 | 41 / 53 | 158 / 200 |

* **The port is the prototype.** The Kokkos element kernel reproduces the host kernel bit for bit on the CPU, for all four functors;
  `adrh_kokkos` against `adrh_proto`: 16 of 18 arrays bitwise equal and the other two (the unmasked runtime-loop kernel, which the
  Kokkos module aliases to the masked one) within 2.2e-16. Against Proteus: element residual and Jacobian 8.6e-16 and 6.0e-16
  (`compare_with_python.py`); the advection-diffusion regression in 2D and 3D and in all seven boundary treatments agrees with the Proteus model
  to 1.2e-14 or better, with the L2 rates of the host module (1.85 to 2.00).
* **Colored assembly is the same bits at every thread count**: 1, 4, 12, 16 on the Mac; 1, 24, 96 on ceds-model; repeated GPU runs
  and both GPUs. It matches a serial scatter to 7e-16 (summation order).
* **Scaling is near-linear in cores** for the element kernel: 11 to 11.6x at 12 threads (the M4 Max has 12 performance cores;
  16 threads, adding the efficiency cores, does not help), 52 to 63x at 96 threads on the Zen 4 node. The fused scatter costs
  7 to 19% more on one thread and 14 to 66% more on 12 to 96 threads, most for the small kernels, where the 30 launches and the
  scattered writes are a larger share.
* **The GPU is not faster than the CPU node here.** One RTX 6000 Ada matches 96 Zen 4 cores on Poisson and the coupled system and is 3x
  slower on the five-component chain (the per-thread arrays are large). That card's double precision rate is a small fraction of its
  single precision rate (spec sheet, not measured here); it is a functional test platform for the device path, not a guide to a data-center GPU.
* **Bitwise results across platforms depend on fused multiply-add.** With contraction on, the CPU native build, the CPU generic
  build and the GPU give three different answers. With it off (`-ffp-contract=off`, `nvcc --fmad=false`) Apple arm64, x86-64 scalar and AVX2,
  and the GPU give **identical bits for the Poisson and five-component chain kernels** (residual and Jacobian), and identical Jacobians for
  all four. The residuals of the two kernels that call `sin` differ in the last bit between the C libraries and CUDA's. So
  `ymf_run --check`-style reproducibility needs the generated kernels built with contraction off, and either no transcendental functions
  in the reproducible path or one implementation of them.

## Building the CUDA benchmark

```
nvcc_wrapper -O3 -std=c++20 -arch=sm_89 -fopenmp --extended-lambda --expt-relaxed-constexpr \
  -I$KOKKOS/include bench_kokkos.cpp -o bench_kokkos_cuda -L$KOKKOS/lib -lkokkoscore -lkokkossimd -lhwloc -lcuda
```

with Kokkos built from source for the card (`-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_ADA89=ON`,
17 seconds): conda-forge's CUDA builds of Kokkos have no sm86 or sm89. The headers compile unchanged under nvcc 13.4; no
device-specific code was added.

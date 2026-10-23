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

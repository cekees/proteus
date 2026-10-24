# ADRH engine prototype

Experimental, not part of the proteus build. A header-only n-component
advection-diffusion-reaction-Hamiltonian element kernel (P1 simplices, Galerkin) with the
coefficient functor as a template parameter, to settle the interface a ymf code generator has to
target. Design: `ADRH_ENGINE_DESIGN.md` (kept beside the campaign notes).

* `adrh.h` – the `Terms<nc,nS>` interface and `elementKernel` (geometry, u and grad u, coefficient
  evaluation and the weak form, residual and Jacobian, all inside the element loop; nothing stored
  per quadrature point). P1 simplices with the affine map, the first version of the kernel.
* `spaces.h`, `engine.h` – Lagrange spaces (`P1Simplex`, `P2Simplex`, `Q1Cube`, in 2 and 3 dimensions) and the
  space-generic `elementKernelSpace`: a space gives basis values and reference gradients at a reference point,
  the map from a face's coordinates to the element's, each face's outward reference normal and vertices; the
  geometry is isoparametric through the same basis at every quadrature point (`PointGeometry`), affine for the
  simplices, trilinear for hexahedra. The masked weak form is the one of `masked.h`.
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
  (`AnalyticalSolutions.LinearADR_Sine`, c = 0) on regular triangles / tets (`AD_SPACE=p1`, `p2`) or quadrilaterals /
  hexahedra (`q1`), in `AD_ND=2` or 3 dimensions: the engine, a plain Proteus model (`ad_sine_p.py`, `ad_sine_n.py`) and the
  exact solution. Engine and Proteus nodal solutions agree to solver round-off (< 5e-14) in every boundary mode, space and
  dimension (42 runs); L2 error rates 2 for P1 and Q1, 3 for P2.

## Boundary conditions (`boundary.h`)

`exteriorFaceKernel` adds the exterior-face terms per face (residual and Jacobian, scattered by the driver), for any
of the spaces (its third template parameter; default P1): segments and triangles of simplices, edges of quadrilaterals,
quadrilaterals of hexahedra, all through the space's own face map and normal. A BC functor says, per component and
boundary quadrature point, which of three kinds applies; each is the engine's version of a Proteus exterior
numerical-flux treatment, and `ad_regression.py` compares against that treatment (`AD_BC=`, in 2D and 3D, for
`AD_SPACE=p1|p2|q1`):

| kind | what it does | Proteus equivalent | `AD_BC` |
|---|---|---|---|
| `Strong` | nothing on the face; the driver replaces the row | `StrongDirichletFactory`, `numericalFluxType=None` | `strong` |
| `Flux` | prescribed diffusive flux, optionally affine in the trace (`gdiff + dgdiff u`); advective flux prescribed where given, else the interior `f(u).n` on outflow and zero on inflow | `StrongDirichletFactory({ci: 'outFlow'})` with `diffusiveFluxBoundaryConditions` / `advectiveFluxBoundaryConditions`; `RobinFactory` for `alpha u + beta q.n = g` | `mixed`, `inflow`, `robin` |
| `WeakDirichlet` | upwinded advective flux against `f(g)`, `-a grad phi . n + (C/h^p)(u - g)`, adjoint `-sigma (u - g) (a^T n).grad w` | `Advection_DiagonalUpwind_Diffusion_{IIPG,NIPG,SIPG}_exterior` (C = 2, 10, 100; sigma = 0, -1, +1) | `iipg`, `nipg`, `sipg` |

The header's comment says where the engine deliberately differs from Proteus (the adjoint's tensor contraction, da/du in
the Jacobian); neither shows with an isotropic constant diffusion tensor.

## Time integration (`ad_dirac_regression.py`)

The element kernel's mass term is backward Euler, `(m(u) - m(u_old)) / dt`, with the coefficients evaluated at the old
state. `ad_dirac_regression.py` steps the transient linear ADR equation `u_t + div(b u - kappa grad u) + c u = 0`
(`TransientLinearADR`, Proteus's `LinearVADR_ConstantCoefficients`) from the Dirac-IC Gaussian
(`AnalyticalSolutions.LinearADR_Decay_DiracIC`, verified in #90) in fixed steps, with the boundary data from the exact
solution at time t through `ExactBC<DiracExact>` in any of the seven `AD_BC` modes, against Proteus's `BackwardEuler` +
`FixedStep` (`ad_dirac_p.py`, `ad_dirac_n.py`) and against the exact solution. `regression_common.py` holds what the two
drivers share. Engine and Proteus agree to round-off (< 1e-13) in every space, dimension and mode; the error against the
exact solution converges at rate 1 in dt and saturates at the time error in h, as backward Euler should.

One Proteus behaviour had to be mimicked for that agreement (`AD_FIRST_STEP=proteus`): Proteus's old mass for the *first*
step is taken from a residual evaluation made after the strongly imposed Dirichlet dofs were already set to their
values at t_1 (`NumericalSolution.calculateSolution`: `initialize_dt_model`, then `getResidual`, then
`stepController.initializeTimeHistory`), so in step one the boundary dofs have no time derivative. The effect is
O(dt^2) and local to the boundary-adjacent rows; later steps are right. The engine's default keeps the true old state.

Design, decisions and numbers: `ADRH_ENGINE_DESIGN.md` (kept beside the campaign notes, sections 11-18).

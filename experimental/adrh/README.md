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
  (`AnalyticalSolutions.LinearADR_Sine`, c = 0) on regular tets: the engine, a plain Proteus model (`ad_sine_p.py`,
  `ad_sine_n.py`) and the exact solution. Engine and Proteus nodal solutions agree to 4e-15; L2 error rates 1.85 and 1.97.
  The driver imposes strong Dirichlet rows; boundary conditions are not in the engine yet.

Design, decisions and numbers: `ADRH_ENGINE_DESIGN.md` (kept beside the campaign notes, sections 11-13).

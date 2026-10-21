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

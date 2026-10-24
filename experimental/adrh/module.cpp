#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <type_traits>
#include "coefficients.h"
#include "boundary.h"
#include "engine.h"
#include <stdexcept>
#include <string>

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <class C, int nS>
py::tuple run(const C &coef, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt)
{
  constexpr int nc = C::nc;
  const int nE = xn.shape(0), nQ = qw.shape(0), nD = nS + 1;
  Arr uo;
  const double *uop = nullptr;
  if (!uold.is_none())
  {
    uo = uold.cast<Arr>();
    uop = uo.data();
  }
  py::array_t<double> R({nE, nc, nD}), J({nE, nc, nD, nc, nD});
  {
    py::gil_scoped_release release;
    adrh::elementKernel<C, nS>(coef, nE, nQ, xn.data(), u.data(), uop, qp.data(), qw.data(), t, dt,
                               R.mutable_data(), J.mutable_data());
  }
  return py::make_tuple(R, J);
}

template <class C, int nS>
py::tuple runMasked(const C &coef, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt)
{
  constexpr int nc = C::nc;
  const int nE = xn.shape(0), nQ = qw.shape(0), nD = nS + 1;
  Arr uo;
  const double *uop = nullptr;
  if (!uold.is_none())
  {
    uo = uold.cast<Arr>();
    uop = uo.data();
  }
  py::array_t<double> R({nE, nc, nD}), J({nE, nc, nD, nc, nD});
  {
    py::gil_scoped_release release;
    adrh::elementKernelMasked<C, nS>(coef, nE, nQ, xn.data(), u.data(), uop, qp.data(), qw.data(), t, dt,
                                     R.mutable_data(), J.mutable_data());
  }
  return py::make_tuple(R, J);
}

// Lagrange spaces by name and dimension; f gets a default-constructed space as a type tag and returns a py::tuple.
template <class F>
py::tuple dispatchSpace(int nd, const std::string &space, F &&f)
{
  if (nd == 2 && space == "p1") return f(adrh::P1Simplex<2>{});
  if (nd == 3 && space == "p1") return f(adrh::P1Simplex<3>{});
  if (nd == 2 && space == "p2") return f(adrh::P2Simplex<2>{});
  if (nd == 3 && space == "p2") return f(adrh::P2Simplex<3>{});
  throw std::invalid_argument("no space '" + space + "' in " + std::to_string(nd) + "D (p1, p2; nd 2 or 3)");
}

PYBIND11_MODULE(adrh_proto, m)
{
  // exterior faces of the advection-diffusion sine problem in nd = 2 (segments) or 3 (triangles) dimensions, for the
  // space "p1" or "p2"; faceKind (nF) per face: 1 = flux face (diffusive flux given, advection by the interior value on
  // outflow), 2 = flux face with the advective flux given too, 3 = weak Dirichlet (penalty_constant / h^penalty_power,
  // adjoint sigma: 0 IIPG, -1 NIPG, +1 SIPG), 4 = Robin alpha u + beta q.n = g with (alpha, beta) given
  m.def("advection_diffusion_sine_faces", [](int nd, const std::string &space, Arr b, double kappa, Arr omega, double omega0,
                                              py::array_t<int, py::array::c_style | py::array::forcecast> faceLocal,
                                              py::array_t<int, py::array::c_style | py::array::forcecast> faceKind,
                                              Arr xn, Arr u, Arr qpf, Arr qwf,
                                              double penalty_constant, double penalty_power, double sigma,
                                              double robin_alpha, double robin_beta) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      adrh::AdvectionDiffusionSine<nS> c(b.data(), kappa, omega.data(), omega0);
      adrh::SineBC<nS> bc(b.data(), kappa, omega.data(), omega0, faceKind.data(), robin_alpha, robin_beta);
      adrh::WeakDirichletParams wd{penalty_constant, penalty_power, sigma};
      const int nF = xn.shape(0), nQf = qwf.shape(0);
      py::array_t<double> Rf({nF, 1, nD}), Jf({nF, 1, nD, 1, nD});
      {
        py::gil_scoped_release release;
        adrh::exteriorFaceKernel<adrh::AdvectionDiffusionSine<nS>, adrh::SineBC<nS>, S>(
            c, bc, wd, nF, nQf, faceLocal.data(), xn.data(), u.data(), qpf.data(), qwf.data(), 0.0,
            Rf.mutable_data(), Jf.mutable_data());
      }
      return py::make_tuple(Rf, Jf);
    });
  });
  // element residual and Jacobian of the same problem, same nd and space
  m.def("advection_diffusion_sine", [](int nd, const std::string &space, Arr b, double kappa, Arr omega, double omega0,
                                        Arr xn, Arr u, Arr qp, Arr qw) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      adrh::AdvectionDiffusionSine<nS> c(b.data(), kappa, omega.data(), omega0);
      const int nE = xn.shape(0), nQ = qw.shape(0);
      py::array_t<double> R({nE, 1, nD}), J({nE, 1, nD, 1, nD});
      {
        py::gil_scoped_release release;
        adrh::elementKernelSpace<adrh::AdvectionDiffusionSine<nS>, S>(c, nE, nQ, xn.data(), u.data(), nullptr,
                                                                      qp.data(), qw.data(), 0.0, 0.0,
                                                                      R.mutable_data(), J.mutable_data());
      }
      return py::make_tuple(R, J);
    });
  });
  // basis values of the space at reference points xi (nQ, nd): (nQ, nDOF), for the driver's error norms
  m.def("basis_values", [](int nd, const std::string &space, Arr xi) {
    return dispatchSpace(nd, space, [&](auto spaceTag) {
      using S = decltype(spaceTag);
      constexpr int nS = S::nS, nD = S::nDOF;
      const int nQ = xi.shape(0);
      py::array_t<double> W({nQ, nD});
      double gw[nD][nS];
      for (int q = 0; q < nQ; ++q) S::basis(xi.data() + q * nS, W.mutable_data() + q * nD, gw);
      return py::make_tuple(W);
    });
  });
  m.def("poisson3d_masked", [](Arr xn, Arr u, Arr qp, Arr qw) {
    return runMasked<adrh::Poisson3D, 3>(adrh::Poisson3D(), xn, u, py::none(), qp, qw, 0.0, 0.0);
  });
  m.def("coupled2_masked", [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt) {
    return runMasked<adrh::Coupled2, 3>(adrh::Coupled2(), xn, u, uold, qp, qw, t, dt);
  });
  // chain5: (masked kernel | runtime-loop kernel) x (sparse masks | all-true masks)
  m.def("chain5", [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt, bool masked, bool sparse) {
    if (masked && sparse) return runMasked<adrh::Chain5<true>, 3>(adrh::Chain5<true>(), xn, u, uold, qp, qw, t, dt);
    if (masked) return runMasked<adrh::Chain5<false>, 3>(adrh::Chain5<false>(), xn, u, uold, qp, qw, t, dt);
    return run<adrh::Chain5<false>, 3>(adrh::Chain5<false>(), xn, u, uold, qp, qw, t, dt);
  });
  m.def("poisson3d", [](Arr xn, Arr u, Arr qp, Arr qw) {
    return run<adrh::Poisson3D, 3>(adrh::Poisson3D(), xn, u, py::none(), qp, qw, 0.0, 0.0);
  });
  m.def("coupled2", [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt) {
    return run<adrh::Coupled2, 3>(adrh::Coupled2(), xn, u, uold, qp, qw, t, dt);
  });
}

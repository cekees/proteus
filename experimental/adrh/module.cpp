#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <type_traits>
#include "coefficients.h"
#include "boundary.h"

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

PYBIND11_MODULE(adrh_proto, m)
{
  // exterior faces of the advection-diffusion sine problem in nd = 2 (segments) or 3 (triangles) dimensions;
  // faceKind (nF) per face: 1 = flux face (diffusive flux given, advection by the interior value on outflow),
  // 2 = flux face with the advective flux given too, 3 = weak Dirichlet (penalty_constant / h^penalty_power,
  // adjoint sigma: 0 IIPG, -1 NIPG, +1 SIPG), 4 = Robin alpha u + beta q.n = g with (alpha, beta) given
  m.def("advection_diffusion_sine_faces", [](int nd, Arr b, double kappa, Arr omega, double omega0,
                                              py::array_t<int, py::array::c_style | py::array::forcecast> faceLocal,
                                              py::array_t<int, py::array::c_style | py::array::forcecast> faceKind,
                                              Arr xn, Arr u, Arr qpf, Arr qwf,
                                              double penalty_constant, double penalty_power, double sigma,
                                              double robin_alpha, double robin_beta) {
    auto run = [&](auto dim) {
      constexpr int nS = decltype(dim)::value;
      adrh::AdvectionDiffusionSine<nS> c(b.data(), kappa, omega.data(), omega0);
      adrh::SineBC<nS> bc(b.data(), kappa, omega.data(), omega0, faceKind.data(), robin_alpha, robin_beta);
      adrh::WeakDirichletParams wd{penalty_constant, penalty_power, sigma};
      const int nF = xn.shape(0), nQf = qwf.shape(0);
      py::array_t<double> Rf({nF, 1, nS + 1}), Jf({nF, 1, nS + 1, 1, nS + 1});
      {
        py::gil_scoped_release release;
        adrh::exteriorFaceKernel<adrh::AdvectionDiffusionSine<nS>, adrh::SineBC<nS>>(
            c, bc, wd, nF, nQf, faceLocal.data(), xn.data(), u.data(), qpf.data(), qwf.data(), 0.0,
            Rf.mutable_data(), Jf.mutable_data());
      }
      return py::make_tuple(Rf, Jf);
    };
    if (nd == 2) return run(std::integral_constant<int, 2>{});
    return run(std::integral_constant<int, 3>{});
  });
  m.def("advection_diffusion_sine", [](int nd, Arr b, double kappa, Arr omega, double omega0, Arr xn, Arr u, Arr qp, Arr qw) {
    if (nd == 2)
    {
      adrh::AdvectionDiffusionSine<2> c(b.data(), kappa, omega.data(), omega0);
      return runMasked<adrh::AdvectionDiffusionSine<2>, 2>(c, xn, u, py::none(), qp, qw, 0.0, 0.0);
    }
    adrh::AdvectionDiffusionSine<3> c(b.data(), kappa, omega.data(), omega0);
    return runMasked<adrh::AdvectionDiffusionSine<3>, 3>(c, xn, u, py::none(), qp, qw, 0.0, 0.0);
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

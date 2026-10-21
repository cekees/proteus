#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "coefficients.h"

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

PYBIND11_MODULE(adrh_proto, m)
{
  m.def("poisson3d", [](Arr xn, Arr u, Arr qp, Arr qw) {
    return run<adrh::Poisson3D, 3>(adrh::Poisson3D(), xn, u, py::none(), qp, qw, 0.0, 0.0);
  });
  m.def("coupled2", [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt) {
    return run<adrh::Coupled2, 3>(adrh::Coupled2(), xn, u, uold, qp, qw, t, dt);
  });
}

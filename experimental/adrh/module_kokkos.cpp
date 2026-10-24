// pybind11 module over the Kokkos element kernel (kokkos_kernel.h), with the host module's calling convention
// (module.cpp), so that golden.py, compare_with_python.py and ad_regression.py can run the same checks on either:
//   ADRH_MODULE=adrh_kokkos python ad_regression.py
// The numpy arrays are copied to the Kokkos execution space and the results copied back, so on a CPU build this
// is the kernel plus a copy, on a device build also the transfers. Threads: KOKKOS_NUM_THREADS (default: all cores).
#define ADRH_USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdlib>
#include <string>
#include <type_traits>

#include "kokkos_kernel.h"

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Mem = Kokkos::DefaultExecutionSpace::memory_space;
using HostView = Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

namespace
{
  adrh::DView<double> up(const Arr &a)
  {
    adrh::DView<double> d("in", a.size());
    Kokkos::deep_copy(d, HostView(const_cast<double *>(a.data()), a.size()));
    return d;
  }

  template <class C, int nS>
  py::tuple run(const C &coef, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt)
  {
    constexpr int nc = C::nc, nD = nS + 1;
    const int nE = (int)xn.shape(0), nQ = (int)qw.shape(0);
    py::array_t<double> R({nE, nc, nD}), J({nE, nc, nD, nc, nD});
    {
      py::gil_scoped_release release;
      adrh::DView<double> dxn = up(xn), du = up(u), dqp = up(qp), dqw = up(qw), duo("uo", 0);
      if (!uold.is_none() && dt > 0.0)
      {
        py::gil_scoped_acquire acquire;
        Arr uo = uold.cast<Arr>();
        duo = up(uo);
      }
      adrh::DView<double> dR("R", R.size()), dJ("J", J.size());
      adrh::elementKernelKokkos<C, nS>(coef, nE, nQ, dxn, du, duo, dqp, dqw, t, dt, dR, dJ);
      Kokkos::deep_copy(HostView(R.mutable_data(), R.size()), dR);
      Kokkos::deep_copy(HostView(J.mutable_data(), J.size()), dJ);
    }
    return py::make_tuple(R, J);
  }

  void finalize()
  {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  }
} // namespace

PYBIND11_MODULE(adrh_kokkos, m)
{
  if (!Kokkos::is_initialized())
  {
    Kokkos::InitializationSettings settings;
    if (const char *n = std::getenv("KOKKOS_NUM_THREADS")) settings.set_num_threads(std::atoi(n));
    Kokkos::initialize(settings);
    py::module_::import("atexit").attr("register")(py::cpp_function(&finalize));
  }
  m.def("kokkos_info", [] {
    py::dict d;
    d["space"] = std::string(Kokkos::DefaultExecutionSpace::name());
    d["concurrency"] = (int)Kokkos::DefaultExecutionSpace().concurrency();
    return d;
  });
  // The masked kernel is what is ported; the unmasked names are aliases so the host module's scripts run unchanged.
  auto poisson = [](Arr xn, Arr u, Arr qp, Arr qw) {
    return run<adrh::Poisson3D, 3>(adrh::Poisson3D(), xn, u, py::none(), qp, qw, 0.0, 0.0);
  };
  auto coupled = [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt) {
    return run<adrh::Coupled2, 3>(adrh::Coupled2(), xn, u, uold, qp, qw, t, dt);
  };
  m.def("poisson3d", poisson);
  m.def("poisson3d_masked", poisson);
  m.def("coupled2", coupled);
  m.def("coupled2_masked", coupled);
  m.def("chain5", [](Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt, bool masked, bool sparse) {
    if (!masked) { PyErr_SetString(PyExc_NotImplementedError, "the Kokkos port is of the masked kernel"); throw py::error_already_set(); }
    if (sparse) return run<adrh::Chain5<true>, 3>(adrh::Chain5<true>(), xn, u, uold, qp, qw, t, dt);
    return run<adrh::Chain5<false>, 3>(adrh::Chain5<false>(), xn, u, uold, qp, qw, t, dt);
  });
  m.def("advection_diffusion_sine", [](int nd, Arr b, double kappa, Arr omega, double omega0, Arr xn, Arr u, Arr qp, Arr qw) {
    if (nd == 2)
    {
      adrh::AdvectionDiffusionSine<2> c(b.data(), kappa, omega.data(), omega0);
      return run<adrh::AdvectionDiffusionSine<2>, 2>(c, xn, u, py::none(), qp, qw, 0.0, 0.0);
    }
    adrh::AdvectionDiffusionSine<3> c(b.data(), kappa, omega.data(), omega0);
    return run<adrh::AdvectionDiffusionSine<3>, 3>(c, xn, u, py::none(), qp, qw, 0.0, 0.0);
  });
}

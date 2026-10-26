// pybind11 module over the Kokkos element kernel (kokkos_kernel.h), with the host module's calling convention
// (module.cpp), so that golden.py, compare_with_python.py and ad_regression.py can run the same checks on either:
//   ADRH_MODULE=adrh_kokkos python ad_regression.py
// The numpy arrays are copied to the Kokkos execution space and the results copied back, so on a CPU build this
// is the kernel plus a copy, on a device build also the transfers. Threads: KOKKOS_NUM_THREADS (default: all cores).
#define ADRH_USE_KOKKOS
#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>

#include "kokkos_kernel.h"
#include "quad_tet4.h"

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Mem = Kokkos::DefaultExecutionSpace::memory_space;
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;
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


  double now()
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  // Colored assembly of the 3-D steady advection-diffusion problem (AdvectionDiffusionSine) on a tetrahedral mesh held in
  // the Kokkos execution space: the coloring and block-CSR pattern are built once on the host, the nodal coordinates and
  // quadrature live on the device, and assemble() returns the global residual and the CSR values to the host, timing the
  // host-to-device copy of the nodal vector, the assembly (best of `repeats`) and the device-to-host copy separately.
  // close() releases the device memory; call it before the interpreter exits (Kokkos is finalized at exit).
  class AdSineAssembler3D
  {
  public:
    AdSineAssembler3D(Arr X, IArr conn, Arr qp, Arr qw)
    {
      nNodes_ = (int)X.shape(0);
      nE_ = (int)conn.shape(0);
      nQ_ = (int)qw.shape(0);
      std::vector<int> c(conn.data(), conn.data() + conn.size());
      const double t0 = now();
      M_ = adrh::makeColoredMesh(nNodes_, nE_, 4, c, rowptr_, colidx_);
      tSetup_ = now() - t0;
      dX_ = up(X); dqp_ = up(qp); dqw_ = up(qw);
      dU_ = adrh::DView<double>("U", nNodes_);
      dUold_ = adrh::DView<double>("Uold", 0);
      dRg_ = adrh::DView<double>("Rg", nNodes_);
      dJv_ = adrh::DView<double>("Jv", colidx_.size());
    }
    py::dict assemble(Arr b, double kappa, Arr omega, double omega0, Arr U, int repeats)
    {
      adrh::AdvectionDiffusionSine<3> coef(b.data(), kappa, omega.data(), omega0);
      py::array_t<double> R(nNodes_), Jv((py::ssize_t)colidx_.size());
      double tH2D, tAsm = 1e30, tD2H;
      {
        py::gil_scoped_release release;
        double t0 = now();
        Kokkos::deep_copy(dU_, HostView(const_cast<double *>(U.data()), U.size()));
        Kokkos::fence();
        tH2D = now() - t0;
        for (int r = 0; r < repeats; ++r)
        {
          t0 = now();
          adrh::assembleColored<adrh::AdvectionDiffusionSine<3>, 3>(coef, M_, nQ_, dX_, dU_, dUold_, dqp_, dqw_, 0.0, 0.0,
                                                                    dRg_, dJv_);
          tAsm = std::min(tAsm, now() - t0);
        }
        t0 = now();
        Kokkos::deep_copy(HostView(R.mutable_data(), R.size()), dRg_);
        Kokkos::deep_copy(HostView(Jv.mutable_data(), Jv.size()), dJv_);
        Kokkos::fence();
        tD2H = now() - t0;
      }
      py::dict d;
      d["R"] = R; d["values"] = Jv;
      d["t_h2d"] = tH2D; d["t_assemble"] = tAsm; d["t_d2h"] = tD2H;
      return d;
    }
    py::array_t<int> indptr() const { return toArr(rowptr_); }
    py::array_t<int> indices() const { return toArr(colidx_); }
    int nColors() const { return M_.nColors; }
    double tSetup() const { return tSetup_; }
    void close()
    {
      M_ = adrh::ColoredMesh();
      dX_ = dqp_ = dqw_ = dU_ = dUold_ = dRg_ = dJv_ = adrh::DView<double>();
    }

  private:
    static py::array_t<int> toArr(const std::vector<int> &v)
    {
      py::array_t<int> a((py::ssize_t)v.size());
      std::copy(v.begin(), v.end(), a.mutable_data());
      return a;
    }
    int nNodes_ = 0, nE_ = 0, nQ_ = 0;
    double tSetup_ = 0.0;
    adrh::ColoredMesh M_;
    std::vector<int> rowptr_, colidx_;
    adrh::DView<double> dX_, dqp_, dqw_, dU_, dUold_, dRg_, dJv_;
  };

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
  py::class_<AdSineAssembler3D>(m, "AdSineAssembler3D")
      .def(py::init<Arr, IArr, Arr, Arr>(), py::arg("X"), py::arg("conn"), py::arg("qp"), py::arg("qw"))
      .def("assemble", &AdSineAssembler3D::assemble, py::arg("b"), py::arg("kappa"), py::arg("omega"), py::arg("omega0"),
           py::arg("U"), py::arg("repeats") = 3)
      .def_property_readonly("indptr", &AdSineAssembler3D::indptr)
      .def_property_readonly("indices", &AdSineAssembler3D::indices)
      .def_property_readonly("n_colors", &AdSineAssembler3D::nColors)
      .def_property_readonly("t_setup", &AdSineAssembler3D::tSetup)
      .def("close", &AdSineAssembler3D::close);
  m.def("quad_tet4", [] {
    py::array_t<double> qp({adrh::quad_tet4::n, 3}), qw(adrh::quad_tet4::n);
    for (int q = 0; q < adrh::quad_tet4::n; ++q)
    {
      qw.mutable_data()[q] = adrh::quad_tet4::weights[q];
      for (int d = 0; d < 3; ++d) qp.mutable_data()[q * 3 + d] = adrh::quad_tet4::points[q][d];
    }
    return py::make_tuple(qp, qw);
  });
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
  // the host module's signature, with the space; the Kokkos port is of the P1 simplex kernel (masked.h) only
  m.def("advection_diffusion_sine", [](int nd, const std::string &space, Arr b, double kappa, Arr omega, double omega0,
                                        Arr xn, Arr u, Arr qp, Arr qw) {
    if (space != "p1")
    {
      PyErr_SetString(PyExc_NotImplementedError, "the Kokkos port covers the P1 simplex kernel; spaces p2 and q1 (engine.h) are not ported");
      throw py::error_already_set();
    }
    if (nd == 2)
    {
      adrh::AdvectionDiffusionSine<2> c(b.data(), kappa, omega.data(), omega0);
      return run<adrh::AdvectionDiffusionSine<2>, 2>(c, xn, u, py::none(), qp, qw, 0.0, 0.0);
    }
    adrh::AdvectionDiffusionSine<3> c(b.data(), kappa, omega.data(), omega0);
    return run<adrh::AdvectionDiffusionSine<3>, 3>(c, xn, u, py::none(), qp, qw, 0.0, 0.0);
  });
}

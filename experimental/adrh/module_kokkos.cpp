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
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

#include "kokkos_kernel.h"
#include "richards.h"
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

  template <class C, class Space>
  py::tuple runSpace(const C &coef, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt)
  {
    constexpr int nc = C::nc, nD = Space::nDOF;
    const int nE = (int)xn.shape(0), nQ = (int)qw.shape(0);
    py::array_t<double> R({nE, nc, nD}), J({nE, nc, nD, nc, nD});
    Arr uo;
    if (!uold.is_none() && dt > 0.0) uo = uold.cast<Arr>();
    {
      py::gil_scoped_release release;
      adrh::DView<double> dxn = up(xn), du = up(u), dqp = up(qp), dqw = up(qw), duo("uo", 0);
      if (uo.size() > 0) duo = up(uo);
      adrh::DView<double> dR("R", R.size()), dJ("J", J.size());
      adrh::elementKernelKokkosSpace<C, Space>(coef, nE, nQ, dxn, du, duo, dqp, dqw, t, dt, dR, dJ);
      Kokkos::deep_copy(HostView(R.mutable_data(), R.size()), dR);
      Kokkos::deep_copy(HostView(J.mutable_data(), J.size()), dJ);
    }
    return py::make_tuple(R, J);
  }

  adrh::DView<int> upInt(const IArr &a)
  {
    adrh::DView<int> d("in", a.size());
    Kokkos::deep_copy(d, Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                             const_cast<int *>(a.data()), a.size()));
    return d;
  }

  // The exterior-face kernel on the execution space. makeBC gets the device pointer of the per-face kind array.
  template <class C, class Space, class BC, class MakeBC>
  py::tuple runFaces(const C &coef, MakeBC makeBC, const adrh::WeakDirichletParams &wd, IArr faceLocal, IArr faceKind, Arr xn, Arr u,
                     Arr qpf, Arr qwf, double t)
  {
    constexpr int nc = C::nc, nD = Space::nDOF;
    const int nF = (int)xn.shape(0), nQf = (int)qwf.shape(0);
    py::array_t<double> Rf({nF, nc, nD}), Jf({nF, nc, nD, nc, nD});
    {
      py::gil_scoped_release release;
      adrh::DView<int> dl = upInt(faceLocal), dk = upInt(faceKind);
      adrh::DView<double> dxn = up(xn), du = up(u), dqp = up(qpf), dqw = up(qwf);
      adrh::DView<double> dR("Rf", Rf.size()), dJ("Jf", Jf.size());
      BC bc = makeBC(dk.data());
      adrh::exteriorFaceKernelKokkos<C, BC, Space>(coef, bc, wd, nF, nQf, dl, adrh::DView<int>("fe", 0), dxn, du, dqp, dqw, t, dR, dJ);
      Kokkos::deep_copy(HostView(Rf.mutable_data(), Rf.size()), dR);
      Kokkos::deep_copy(HostView(Jf.mutable_data(), Jf.size()), dJ);
    }
    return py::make_tuple(Rf, Jf);
  }

  // Lagrange spaces by name and dimension; f gets a default-constructed space as a type tag and returns a py::tuple.
  template <class F>
  py::tuple dispatchSpace(int nd, const std::string &space, F &&f)
  {
    if (nd == 1 && space == "p1") return f(adrh::P1Simplex<1>{});
    if (nd == 1 && space == "p2") return f(adrh::P2Simplex<1>{});
    if (nd == 2 && space == "p1") return f(adrh::P1Simplex<2>{});
    if (nd == 3 && space == "p1") return f(adrh::P1Simplex<3>{});
    if (nd == 2 && space == "p2") return f(adrh::P2Simplex<2>{});
    if (nd == 3 && space == "p2") return f(adrh::P2Simplex<3>{});
    if (nd == 2 && space == "q1") return f(adrh::Q1Cube<2>{});
    if (nd == 3 && space == "q1") return f(adrh::Q1Cube<3>{});
    throw std::invalid_argument("no space '" + space + "' in " + std::to_string(nd) + "D (p1, p2, q1; nd 1 to 3, q1 only 2 or 3)");
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
  // exterior faces of the sine problem (host module's signature); kind per face: 1 flux, 2 flux with the advective flux, 3 weak Dirichlet,
  // 4 Robin
  m.def("advection_diffusion_sine_faces", [](int nd, const std::string &space, Arr b, double kappa, Arr omega, double omega0, IArr faceLocal,
                                              IArr faceKind, Arr xn, Arr u, Arr qpf, Arr qwf, double penalty_constant,
                                              double penalty_power, double sigma, double robin_alpha, double robin_beta) {
    return dispatchSpace(nd, space, [&](auto tag) {
      using S = decltype(tag);
      constexpr int nS = S::nS;
      using Co = adrh::AdvectionDiffusionSine<nS>;
      using BC = adrh::SineBC<nS>;
      Co c(b.data(), kappa, omega.data(), omega0);
      adrh::WeakDirichletParams wd{penalty_constant, penalty_power, sigma};
      auto make = [&](const int *dk) { return BC(b.data(), kappa, omega.data(), omega0, dk, robin_alpha, robin_beta); };
      return runFaces<Co, S, BC>(c, make, wd, faceLocal, faceKind, xn, u, qpf, qwf, 0.0);
    });
  });
  // the transient linear advection-diffusion-reaction element kernel (backward Euler mass term), any space
  m.def("transient_adr", [](int nd, const std::string &space, Arr b, double kappa, double c, Arr xn, Arr u, Arr uold, Arr qp, Arr qw,
                             double t, double dt) {
    return dispatchSpace(nd, space, [&](auto tag) {
      using S = decltype(tag);
      adrh::TransientLinearADR<S::nS> co(b.data(), kappa, c);
      return runSpace<adrh::TransientLinearADR<S::nS>, S>(co, xn, u, uold, qp, qw, t, dt);
    });
  });
  // its exterior faces, with boundary data from the Dirac-IC Gaussian
  m.def("transient_adr_faces", [](int nd, const std::string &space, Arr b, double kappa, double c, double tStart, double u0, Arr x0,
                                   IArr faceLocal, IArr faceKind, Arr xn, Arr u, Arr qpf, Arr qwf, double t, double penalty_constant,
                                   double penalty_power, double sigma, double robin_alpha, double robin_beta) {
    return dispatchSpace(nd, space, [&](auto tag) {
      using S = decltype(tag);
      constexpr int nS = S::nS;
      using Co = adrh::TransientLinearADR<nS>;
      using BC = adrh::ExactBC<nS, adrh::DiracExact<nS>>;
      Co co(b.data(), kappa, c);
      adrh::DiracExact<nS> exact(b.data(), kappa, c, tStart, u0, x0.data());
      adrh::WeakDirichletParams wd{penalty_constant, penalty_power, sigma};
      auto make = [&](const int *dk) { return BC(exact, b.data(), kappa, dk, robin_alpha, robin_beta); };
      return runFaces<Co, S, BC>(co, make, wd, faceLocal, faceKind, xn, u, qpf, qwf, t);
    });
  });
  // Richards' equation (richards.h) with several materials; see module.cpp for the arguments
  m.def("richards_vgm", [](int nd, const std::string &space, Arr alpha, Arr n_vg, Arr thetaR, Arr thetaSR, Arr Ks, Arr g, double rho,
                            double beta, IArr material, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt) {
    return dispatchSpace(nd, space, [&](auto tag) {
      using S = decltype(tag);
      using Co = adrh::RichardsMaterials<S::nS>;
      constexpr int nc = 1, nD = S::nDOF;
      const int nE = (int)xn.shape(0), nQ = (int)qw.shape(0);
      py::array_t<double> R({nE, nc, nD}), J({nE, nc, nD, nc, nD});
      Arr uo;
      if (!uold.is_none() && dt > 0.0) uo = uold.cast<Arr>();
      {
        py::gil_scoped_release release;
        adrh::DView<int> dmat = upInt(material);
        Co co((int)alpha.size(), alpha.data(), n_vg.data(), thetaR.data(), thetaSR.data(), Ks.data(), g.data(), dmat.data(), rho, beta);
        adrh::DView<double> dxn = up(xn), du = up(u), dqp = up(qp), dqw = up(qw), duo("uo", 0);
        if (uo.size() > 0) duo = up(uo);
        adrh::DView<double> dR("R", R.size()), dJ("J", J.size());
        adrh::elementKernelKokkosSpace<Co, S>(co, nE, nQ, dxn, du, duo, dqp, dqw, t, dt, dR, dJ);
        Kokkos::deep_copy(HostView(R.mutable_data(), R.size()), dR);
        Kokkos::deep_copy(HostView(J.mutable_data(), J.size()), dJ);
      }
      return py::make_tuple(R, J);
    });
  });
  m.def("richards_vgm_faces", [](int nd, const std::string &space, Arr alpha, Arr n_vg, Arr thetaR, Arr thetaSR, Arr Ks, Arr g, double rho,
                                  double beta, IArr material, IArr faceElem, IArr faceLocal, Arr gD, Arr xn, Arr u, Arr qpf, Arr qwf,
                                  double t, double penalty_constant, double penalty_power) {
    return dispatchSpace(nd, space, [&](auto tag) {
      using S = decltype(tag);
      using Co = adrh::RichardsMaterials<S::nS>;
      adrh::WeakDirichletParams wd{penalty_constant, penalty_power, 0.0};
      constexpr int nc = 1, nD = S::nDOF;
      const int nF = (int)xn.shape(0), nQf = (int)qwf.shape(0);
      py::array_t<double> Rf({nF, nc, nD}), Jf({nF, nc, nD, nc, nD});
      {
        py::gil_scoped_release release;
        adrh::DView<int> dmat = upInt(material), dl = upInt(faceLocal), de = upInt(faceElem);
        Co co((int)alpha.size(), alpha.data(), n_vg.data(), thetaR.data(), thetaSR.data(), Ks.data(), g.data(), dmat.data(), rho, beta);
        adrh::DView<double> dg = up(gD), dxn = up(xn), du = up(u), dqp = up(qpf), dqw = up(qwf);
        adrh::DView<double> dR("Rf", Rf.size()), dJ("Jf", Jf.size());
        adrh::FixedValueBC bc(dg.data());
        adrh::exteriorFaceKernelKokkos<Co, adrh::FixedValueBC, S>(co, bc, wd, nF, nQf, dl, de, dxn, du, dqp, dqw, t, dR, dJ);
        Kokkos::deep_copy(HostView(Rf.mutable_data(), Rf.size()), dR);
        Kokkos::deep_copy(HostView(Jf.mutable_data(), Jf.size()), dJ);
      }
      return py::make_tuple(Rf, Jf);
    });
  });
  // ---- algebraic stabilization (lowOrder.h), on the execution space ----
  m.def("transpose_map", [](IArr indptr, IArr indices) {
    const std::vector<int> t = adrh::buildTransposeMap((int)indptr.size() - 1, indptr.data(), indices.data());
    py::array_t<int> out((py::ssize_t)t.size());
    std::copy(t.begin(), t.end(), out.mutable_data());
    return out;
  });
  m.def("algebraic_dissipation", [](IArr indptr, IArr indices, IArr tpos, Arr values, Arr u, Arr r) {
    py::array_t<double> J(values.size()), R(r.size());
    {
      py::gil_scoped_release release;
      adrh::DView<int> dp = upInt(indptr), di = upInt(indices), dt = upInt(tpos);
      adrh::DView<double> dJ = up(values), du = up(u), dR = up(r);
      adrh::algebraicDissipationKokkos((int)indptr.size() - 1, dp, di, dt, dJ, du, dR);
      Kokkos::deep_copy(HostView(J.mutable_data(), J.size()), dJ);
      Kokkos::deep_copy(HostView(R.mutable_data(), R.size()), dR);
    }
    return py::make_tuple(J, R);
  });
  // the same pass with the three phases timed: returns (values, r, {t_h2d, t_pass, t_d2h}); the pass is best of `repeats` (pass 1 reads the
  // original values each time, so the device copy is refreshed between repeats outside the timing)
  m.def("algebraic_dissipation_timed", [](IArr indptr, IArr indices, IArr tpos, Arr values, Arr u, Arr r, int repeats) {
    py::array_t<double> J(values.size()), R(r.size());
    double tH = 0, tP = 1e30, tD = 0;
    {
      py::gil_scoped_release release;
      adrh::DView<int> dp = upInt(indptr), di = upInt(indices), dt = upInt(tpos);
      double t0 = now();
      adrh::DView<double> dJ = up(values), du = up(u), dR = up(r), dJ0 = up(values), dR0 = up(r);
      Kokkos::fence();
      tH = now() - t0;
      for (int k = 0; k < repeats; ++k)
      {
        Kokkos::deep_copy(dJ, dJ0); Kokkos::deep_copy(dR, dR0); Kokkos::fence();
        t0 = now();
        adrh::algebraicDissipationKokkos((int)indptr.size() - 1, dp, di, dt, dJ, du, dR);
        tP = std::min(tP, now() - t0);
      }
      t0 = now();
      Kokkos::deep_copy(HostView(J.mutable_data(), J.size()), dJ);
      Kokkos::deep_copy(HostView(R.mutable_data(), R.size()), dR);
      Kokkos::fence();
      tD = now() - t0;
    }
    py::dict d; d["t_h2d"] = tH; d["t_pass"] = tP; d["t_d2h"] = tD;
    return py::make_tuple(J, R, d);
  });
  m.def("richards_vgm_lumped", [](int nd, const std::string &space, Arr alpha, Arr n_vg, Arr thetaR, Arr thetaSR, Arr Ks, Arr g, double rho,
                                   double beta, IArr material, Arr xn, Arr u, py::object uold, Arr qp, Arr qw, double t, double dt) {
    return dispatchSpace(nd, space, [&](auto tag) {
      using S = decltype(tag);
      using Co = adrh::RichardsMaterials<S::nS>;
      constexpr int nD = S::nDOF;
      const int nE = (int)xn.shape(0), nQ = (int)qw.shape(0);
      py::array_t<double> w({nE, nD}), mm({nE, nD}), dm({nE, nD}), mold({nE, nD});
      Arr uo;
      if (!uold.is_none() && dt > 0.0) uo = uold.cast<Arr>();
      {
        py::gil_scoped_release release;
        adrh::DView<int> dmat = upInt(material);
        Co co((int)alpha.size(), alpha.data(), n_vg.data(), thetaR.data(), thetaSR.data(), Ks.data(), g.data(), dmat.data(), rho, beta);
        adrh::DView<double> dxn = up(xn), du = up(u), dqp = up(qp), dqw = up(qw), duo("uo", 0);
        if (uo.size() > 0) duo = up(uo);
        adrh::DView<double> dw("w", w.size()), dmm("m", mm.size()), ddm("dm", dm.size()), dmo("mold", mold.size());
        adrh::lumpedMassKokkos<Co, S>(co, nE, nQ, dxn, du, duo, dqp, dqw, t, dt, dw, dmm, ddm, dmo);
        Kokkos::deep_copy(HostView(w.mutable_data(), w.size()), dw);
        Kokkos::deep_copy(HostView(mm.mutable_data(), mm.size()), dmm);
        Kokkos::deep_copy(HostView(dm.mutable_data(), dm.size()), ddm);
        Kokkos::deep_copy(HostView(mold.mutable_data(), mold.size()), dmo);
      }
      return py::make_tuple(w, mm, dm, mold);
    });
  });
  m.def("transient_adr_lumped", [](int nd, const std::string &space, Arr b, double kappa, double c, Arr xn, Arr u, py::object uold, Arr qp,
                                    Arr qw, double t, double dt) {
    return dispatchSpace(nd, space, [&](auto tag) {
      using S = decltype(tag);
      using Co = adrh::TransientLinearADR<S::nS>;
      constexpr int nD = S::nDOF;
      Co co(b.data(), kappa, c);
      const int nE = (int)xn.shape(0), nQ = (int)qw.shape(0);
      py::array_t<double> w({nE, nD}), mm({nE, nD}), dm({nE, nD}), mold({nE, nD});
      Arr uo;
      if (!uold.is_none() && dt > 0.0) uo = uold.cast<Arr>();
      {
        py::gil_scoped_release release;
        adrh::DView<double> dxn = up(xn), du = up(u), dqp = up(qp), dqw = up(qw), duo("uo", 0);
        if (uo.size() > 0) duo = up(uo);
        adrh::DView<double> dw("w", w.size()), dmm("m", mm.size()), ddm("dm", dm.size()), dmo("mold", mold.size());
        adrh::lumpedMassKokkos<Co, S>(co, nE, nQ, dxn, du, duo, dqp, dqw, t, dt, dw, dmm, ddm, dmo);
        Kokkos::deep_copy(HostView(w.mutable_data(), w.size()), dw);
        Kokkos::deep_copy(HostView(mm.mutable_data(), mm.size()), dmm);
        Kokkos::deep_copy(HostView(dm.mutable_data(), dm.size()), ddm);
        Kokkos::deep_copy(HostView(mold.mutable_data(), mold.size()), dmo);
      }
      return py::make_tuple(w, mm, dm, mold);
    });
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
  // the host module's signature, with the space: p1 runs the affine simplex kernel (masked.h), p2 and q1 the Lagrange-space kernel
  // (engine.h, isoparametric geometry at every point)
  m.def("advection_diffusion_sine", [](int nd, const std::string &space, Arr b, double kappa, Arr omega, double omega0,
                                        Arr xn, Arr u, Arr qp, Arr qw) -> py::tuple {
    if (space == "p2" || space == "q1")
    {
      auto go = [&](auto tag) -> py::tuple {
        using S = decltype(tag);
        adrh::AdvectionDiffusionSine<S::nS> c(b.data(), kappa, omega.data(), omega0);
        return runSpace<adrh::AdvectionDiffusionSine<S::nS>, S>(c, xn, u, py::none(), qp, qw, 0.0, 0.0);
      };
      if (space == "p2") return nd == 2 ? go(adrh::P2Simplex<2>{}) : go(adrh::P2Simplex<3>{});
      return nd == 2 ? go(adrh::Q1Cube<2>{}) : go(adrh::Q1Cube<3>{});
    }
    if (space != "p1")
    {
      PyErr_SetString(PyExc_ValueError, "space must be p1, p2 or q1");
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

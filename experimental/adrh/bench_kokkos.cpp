// Benchmark and check of the Kokkos port of the ADRH element kernel (kokkos_kernel.h), on a structured
// tetrahedral mesh (the unit cube, 6 Kuhn tetrahedra per cell, nodes perturbed so no element is a copy of another).
//
// For each coefficient functor (Poisson, a coupled two-component system with mass terms, a five-component chain,
// advection-diffusion with the manufactured sine solution) it
//   1. runs the host prototype (elementKernelMasked, serial) for the reference residual and Jacobian,
//   2. runs elementKernelKokkos and compares it with the reference (bitwise where the arithmetic is the same),
//   3. runs the fused colored assembly and compares it with a serial scatter of the reference,
//   4. prints the time per element of each and a checksum of the colored result's bits, which must not change
//      with the number of threads (or between Kokkos backends that do the same arithmetic).
//
// usage: bench_kokkos [nn] [repeats] [--kokkos-num-threads=N]      (nn nodes per edge; nn=31 is 162,000 tets)
#define ADRH_USE_KOKKOS
#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kokkos_kernel.h"
#include "quad_tet4.h"

using namespace adrh;

struct Tets
{
  int nNodes = 0, nE = 0;
  std::vector<double> X;   // (nNodes, 3)
  std::vector<int> conn;   // (nE, 4)
};

static Tets cubeTets(int nn)
{
  Tets m;
  m.nNodes = nn * nn * nn;
  m.X.resize(m.nNodes * 3);
  const double h = 1.0 / (nn - 1);
  auto id = [&](int i, int j, int k) { return i + nn * (j + nn * k); };
  auto jitter = [](long n, int d) {   // deterministic, in [-1, 1]
    uint64_t x = (uint64_t)n * 6364136223846793005ULL + (uint64_t)d * 1442695040888963407ULL + 12345;
    x ^= x >> 29; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 32;
    return ((x >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
  };
  for (int k = 0; k < nn; ++k)
    for (int j = 0; j < nn; ++j)
      for (int i = 0; i < nn; ++i)
      {
        const int n = id(i, j, k);
        const int p[3] = {i, j, k};
        for (int d = 0; d < 3; ++d)
        {
          const bool boundary = p[d] == 0 || p[d] == nn - 1;
          m.X[n * 3 + d] = p[d] * h + (boundary ? 0.0 : 0.15 * h * jitter(n, d));
        }
      }
  const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int k = 0; k < nn - 1; ++k)
    for (int j = 0; j < nn - 1; ++j)
      for (int i = 0; i < nn - 1; ++i)
        for (int p = 0; p < 6; ++p)
        {
          int c = 0, off[3] = {0, 0, 0};
          int nodes[4];
          nodes[0] = id(i, j, k);
          for (int s = 0; s < 3; ++s)
          {
            off[perm[p][s]] = 1;
            nodes[s + 1] = id(i + off[0], j + off[1], k + off[2]);
          }
          (void)c;
          for (int a = 0; a < 4; ++a) m.conn.push_back(nodes[a]);
        }
  m.nE = (int)(m.conn.size() / 4);
  return m;
}

static double seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint64_t fnv(const std::vector<double> &v)
{
  uint64_t h = 1469598103934665603ULL;
  for (double d : v)
  {
    uint64_t bits;
    std::memcpy(&bits, &d, 8);
    h = (h ^ bits) * 1099511628211ULL;
  }
  return h;
}

static double relDiff(const std::vector<double> &a, const std::vector<double> &b)
{
  double d = 0.0, s = 1e-300;
  for (size_t k = 0; k < a.size(); ++k) { d = std::max(d, std::fabs(a[k] - b[k])); s = std::max(s, std::fabs(b[k])); }
  return d / s;
}

template <class T>
static DView<T> toDevice(const char *name, const std::vector<T> &h)
{
  DView<T> d(name, h.size());
  auto hv = Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h.data(), h.size());
  Kokkos::deep_copy(d, hv);
  return d;
}

template <class T>
static std::vector<T> toHost(DView<T> d)
{
  std::vector<T> h(d.extent(0));
  Kokkos::deep_copy(Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h.data(), h.size()), d);
  return h;
}

template <class F>
static double best(int repeats, F f)
{
  double b = 1e30;
  for (int r = 0; r < repeats; ++r) { const double t0 = seconds(); f(); b = std::min(b, seconds() - t0); }
  return b;
}

template <class C>
static void runCase(const char *name, const C &coef, const Tets &m, bool unsteady, int repeats)
{
  constexpr int nS = 3, nD = 4, nc = C::nc;
  const int nE = m.nE, nQ = quad_tet4::n;
  const double t = 0.7, dt = unsteady ? 0.05 : 0.0;

  std::vector<double> qp(nQ * 3), qw(nQ);
  for (int q = 0; q < nQ; ++q) { qw[q] = quad_tet4::weights[q]; for (int d = 0; d < 3; ++d) qp[q * 3 + d] = quad_tet4::points[q][d]; }

  // nodal values, deterministic in [0.2, 1]
  std::vector<double> U(m.nNodes * nc), Uo(unsteady ? m.nNodes * nc : 0);
  uint64_t s = 88172645463325252ULL;
  auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return 0.2 + 0.8 * ((s >> 11) * (1.0 / 9007199254740992.0)); };
  for (auto &v : U) v = rnd();
  for (auto &v : Uo) v = rnd();

  // element-local arrays in the prototype's layout
  std::vector<double> xn((size_t)nE * nD * 3), ue((size_t)nE * nc * nD), uo(unsteady ? (size_t)nE * nc * nD : 0);
  for (int e = 0; e < nE; ++e)
    for (int a = 0; a < nD; ++a)
    {
      const int n = m.conn[e * nD + a];
      for (int d = 0; d < 3; ++d) xn[((size_t)e * nD + a) * 3 + d] = m.X[n * 3 + d];
      for (int j = 0; j < nc; ++j)
      {
        ue[((size_t)e * nc + j) * nD + a] = U[n * nc + j];
        if (unsteady) uo[((size_t)e * nc + j) * nD + a] = Uo[n * nc + j];
      }
    }

  // 1. the host prototype
  std::vector<double> Rref((size_t)nE * nc * nD), Jref((size_t)nE * nc * nD * nc * nD);
  const double tHost = best(repeats > 2 ? 2 : repeats, [&] {
    elementKernelMasked<C, nS>(coef, nE, nQ, xn.data(), ue.data(), unsteady ? uo.data() : nullptr, qp.data(), qw.data(),
                               t, dt, Rref.data(), Jref.data());
  });

  // the serial scatter of the reference, in element order
  std::vector<int> rowptr, colidx;
  ColoredMesh M = makeColoredMesh(m.nNodes, nE, nD, m.conn, rowptr, colidx);
  std::vector<int> blockOf = toHost(M.blockOf);
  std::vector<double> Rg_ref((size_t)m.nNodes * nc, 0.0), Jv_ref((size_t)M.nBlocks * nc * nc, 0.0);
  for (int e = 0; e < nE; ++e)
    for (int a = 0; a < nD; ++a)
    {
      const int n = m.conn[e * nD + a];
      for (int i = 0; i < nc; ++i) Rg_ref[n * nc + i] += Rref[((size_t)e * nc + i) * nD + a];
      for (int b = 0; b < nD; ++b)
      {
        double *blk = &Jv_ref[(size_t)blockOf[((size_t)e * nD + a) * nD + b] * nc * nc];
        for (int i = 0; i < nc; ++i)
          for (int j = 0; j < nc; ++j) blk[i * nc + j] += Jref[((((size_t)e * nc + i) * nD + a) * nc + j) * nD + b];
      }
    }

  // 2. the Kokkos element kernel
  DView<double> dxn = toDevice("xn", xn), due = toDevice("u", ue), duo = unsteady ? toDevice("uo", uo) : DView<double>("uo", 0);
  DView<double> dqp = toDevice("qp", qp), dqw = toDevice("qw", qw);
  DView<double> dR("R", Rref.size()), dJ("J", Jref.size());
  auto run1 = [&] {
    elementKernelKokkos<C, nS>(coef, nE, nQ, dxn, due, duo, dqp, dqw, t, dt, dR, dJ);
  };
  run1();   // warm up (and compile on first use)
  const double tK = best(repeats, run1);
  std::vector<double> Rk = toHost(dR), Jk = toHost(dJ);
  const bool bitwise = Rk == Rref && Jk == Jref;

  // 3. the fused colored assembly
  DView<double> dX = toDevice("X", m.X), dU = toDevice("U", U), dUo = unsteady ? toDevice("Uo", Uo) : DView<double>("Uo", 0);
  DView<double> dRg("Rg", Rg_ref.size()), dJv("Jv", Jv_ref.size());
  auto run2 = [&] {
    assembleColored<C, nS>(coef, M, nQ, dX, dU, dUo, dqp, dqw, t, dt, dRg, dJv);
  };
  run2();
  const double tC = best(repeats, run2);
  std::vector<double> Rg = toHost(dRg), Jv = toHost(dJv);

  std::printf("%-22s %-7s thr=%-3d nE=%-8d | element kernel %7.1f ns/el (host serial %7.1f, x%.1f) vs prototype: R %.1e J %.1e %s"
              " | colored assembly %7.1f ns/el (%d colors) vs serial scatter: R %.1e J %.1e  checksum %016llx %016llx\n",
              name, Kokkos::DefaultExecutionSpace::name(), (int)Kokkos::DefaultExecutionSpace().concurrency(), nE,
              1e9 * tK / nE, 1e9 * tHost / nE, tHost / tK, relDiff(Rk, Rref), relDiff(Jk, Jref),
              bitwise ? "bitwise" : "       ", 1e9 * tC / nE, M.nColors, relDiff(Rg, Rg_ref), relDiff(Jv, Jv_ref),
              (unsigned long long)fnv(Rg), (unsigned long long)fnv(Jv));
}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    int nn = 31, repeats = 5;
    int pos = 0;
    for (int a = 1; a < argc; ++a)
    {
      if (argv[a][0] == '-') continue;
      (pos++ == 0 ? nn : repeats) = std::atoi(argv[a]);
    }
    const Tets m = cubeTets(nn);
    const double b[3] = {1.0, 0.5, 0.25}, om[3] = {3.0, 2.0, 4.0};
    runCase("poisson3d (nc=1)", Poisson3D(), m, false, repeats);
    runCase("ad_sine (nc=1)", AdvectionDiffusionSine<3>(b, 0.2, om, 0.3), m, false, repeats);
    runCase("coupled2 (nc=2, mass)", Coupled2(), m, true, repeats);
    runCase("chain5 (nc=5, mass)", Chain5<true>(), m, true, repeats);
  }
  Kokkos::finalize();
  return 0;
}

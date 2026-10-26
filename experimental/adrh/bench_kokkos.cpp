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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "kokkos_kernel.h"
#include "quad_tet4.h"

using namespace adrh;

struct Tets
{
  int nNodes = 0, nE = 0, nD = 4;
  std::vector<double> X;   // (nNodes, 3): every DOF's coordinates
  std::vector<int> conn;   // (nE, nD)
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


// The same Kuhn tetrahedra with quadratic DOFs: the 4 vertices, then the 6 edge midpoints in proteus's order
// (0,1), (1,2), (2,3), (0,2), (1,3), (0,3); a midpoint is one DOF shared by every element that has the edge.
static Tets cubeTetsP2(int nn)
{
  Tets p1 = cubeTets(nn), m;
  m.nNodes = p1.nNodes; m.nE = p1.nE; m.nD = 10; m.X = p1.X;
  m.conn.resize((size_t)m.nE * 10);
  const int edges[6][2] = {{0, 1}, {1, 2}, {2, 3}, {0, 2}, {1, 3}, {0, 3}};
  std::map<std::pair<int, int>, int> mid;
  for (int e = 0; e < m.nE; ++e)
  {
    for (int a = 0; a < 4; ++a) m.conn[e * 10 + a] = p1.conn[e * 4 + a];
    for (int k = 0; k < 6; ++k)
    {
      int a = p1.conn[e * 4 + edges[k][0]], b = p1.conn[e * 4 + edges[k][1]];
      if (a > b) std::swap(a, b);
      auto it = mid.find({a, b});
      if (it == mid.end())
      {
        it = mid.emplace(std::make_pair(a, b), m.nNodes++).first;
        for (int d = 0; d < 3; ++d) m.X.push_back(0.5 * (m.X[a * 3 + d] + m.X[b * 3 + d]));
      }
      m.conn[e * 10 + 4 + k] = it->second;
    }
  }
  return m;
}

// Trilinear hexahedra of the unit cube, nodes in proteus's order: the square (-,-), (+,-), (+,+), (-,+) at the low z, then at the high z.
static Tets cubeHexes(int nn)
{
  Tets t = cubeTets(nn), m;                   // the node coordinates (jittered, so no element is a copy of another)
  m.nNodes = t.nNodes; m.X = t.X; m.nD = 8; m.nE = (nn - 1) * (nn - 1) * (nn - 1);
  auto id = [&](int i, int j, int k) { return i + nn * (j + nn * k); };
  const int corner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  for (int k = 0; k < nn - 1; ++k)
    for (int j = 0; j < nn - 1; ++j)
      for (int i = 0; i < nn - 1; ++i)
        for (int c = 0; c < 8; ++c) m.conn.push_back(id(i + corner[c][0], j + corner[c][1], k + corner[c][2]));
  return m;
}

// Gauss-Legendre, 3 points per axis, on [-1, 1]^3 (proteus's cube)
static void cubeGauss(std::vector<double> &qp, std::vector<double> &qw)
{
  const double g[3] = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)}, w[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
      {
        qp.push_back(g[i]); qp.push_back(g[j]); qp.push_back(g[k]);
        qw.push_back(w[i] * w[j] * w[k]);
      }
}

static void tetQuadrature(std::vector<double> &qp, std::vector<double> &qw)
{
  for (int q = 0; q < quad_tet4::n; ++q)
  {
    qw.push_back(quad_tet4::weights[q]);
    for (int d = 0; d < 3; ++d) qp.push_back(quad_tet4::points[q][d]);
  }
}

// What differs between the three element kinds in the runner: the host reference and the two Kokkos launches.
template <class C, int nS_>
struct KindP1
{
  static constexpr int nS = nS_, nD = nS_ + 1;
  static void host(const C &c, int nE, int nQ, const double *xn, const double *u, const double *uo, const double *qp,
                   const double *qw, double t, double dt, double *R, double *J)
  { elementKernelMasked<C, nS_>(c, nE, nQ, xn, u, uo, qp, qw, t, dt, R, J); }
  template <class... A> static void element(A &&... a) { elementKernelKokkos<C, nS_>(std::forward<A>(a)...); }
  template <class... A> static void colored(A &&... a) { assembleColored<C, nS_>(std::forward<A>(a)...); }
};
template <class C, class Space>
struct KindSpace
{
  static constexpr int nS = Space::nS, nD = Space::nDOF;
  static void host(const C &c, int nE, int nQ, const double *xn, const double *u, const double *uo, const double *qp,
                   const double *qw, double t, double dt, double *R, double *J)
  { elementKernelSpace<C, Space>(c, nE, nQ, xn, u, uo, qp, qw, t, dt, R, J); }
  template <class... A> static void element(A &&... a) { elementKernelKokkosSpace<C, Space>(std::forward<A>(a)...); }
  template <class... A> static void colored(A &&... a) { assembleColoredSpace<C, Space>(std::forward<A>(a)...); }
};

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

template <class C, class Kind>
static void runCase(const char *name, const C &coef, const Tets &m, const std::vector<double> &qp, const std::vector<double> &qw,
                    bool unsteady, int repeats)
{
  constexpr int nD = Kind::nD, nc = C::nc;
  const int nE = m.nE, nQ = (int)qw.size();
  const double t = 0.7, dt = unsteady ? 0.05 : 0.0;

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
    Kind::host(coef, nE, nQ, xn.data(), ue.data(), unsteady ? uo.data() : nullptr, qp.data(), qw.data(), t, dt, Rref.data(),
               Jref.data());
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
    Kind::element(coef, nE, nQ, dxn, due, duo, dqp, dqw, t, dt, dR, dJ);
  };
  run1();   // warm up (and compile on first use)
  const double tK = best(repeats, run1);
  std::vector<double> Rk = toHost(dR), Jk = toHost(dJ);
  const bool bitwise = Rk == Rref && Jk == Jref;

  // 3. the fused colored assembly
  DView<double> dX = toDevice("X", m.X), dU = toDevice("U", U), dUo = unsteady ? toDevice("Uo", Uo) : DView<double>("Uo", 0);
  DView<double> dRg("Rg", Rg_ref.size()), dJv("Jv", Jv_ref.size());
  auto run2 = [&] {
    Kind::colored(coef, M, nQ, dX, dU, dUo, dqp, dqw, t, dt, dRg, dJv);
  };
  run2();
  const double tC = best(repeats, run2);
  std::vector<double> Rg = toHost(dRg), Jv = toHost(dJv);

  std::printf("%-26s %-7s thr=%-3d nE=%-8d | element kernel %7.1f ns/el (host serial %7.1f, x%.1f) vs prototype: R %.1e J %.1e %s"
              " | colored assembly %7.1f ns/el (%d colors) vs serial scatter: R %.1e J %.1e  checksum %016llx %016llx\n",
              name, Kokkos::DefaultExecutionSpace::name(), (int)Kokkos::DefaultExecutionSpace().concurrency(), nE,
              1e9 * tK / nE, 1e9 * tHost / nE, tHost / tK, relDiff(Rk, Rref), relDiff(Jk, Jref),
              bitwise ? "bitwise" : "       ", 1e9 * tC / nE, M.nColors, relDiff(Rg, Rg_ref), relDiff(Jv, Jv_ref),
              (unsigned long long)fnv(Rg), (unsigned long long)fnv(Jv));
}


// ---- exterior faces ---------------------------------------------------------------------------------------------

// 3-point rule on the reference triangle (area 1/2) and 3 x 3 Gauss on [-1, 1]^2: the face quadratures of tetrahedra and hexahedra
static void faceQuadrature(bool cube, std::vector<double> &qp, std::vector<double> &qw)
{
  if (!cube)
  {
    const double p[3][2] = {{1.0 / 6.0, 1.0 / 6.0}, {2.0 / 3.0, 1.0 / 6.0}, {1.0 / 6.0, 2.0 / 3.0}};
    for (int q = 0; q < 3; ++q) { qp.push_back(p[q][0]); qp.push_back(p[q][1]); qw.push_back(1.0 / 6.0); }
    return;
  }
  const double g[3] = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)}, w[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) { qp.push_back(g[i]); qp.push_back(g[j]); qw.push_back(w[i] * w[j]); }
}

// The sine problem's faces of one space on a mesh: the kernel per face against the host, and the colored scatter of the faces into
// the global residual and block values against a serial scatter of the host results.
template <class Space>
static void runFaceCase(const char *name, const Tets &m, int repeats)
{
  using Co = AdvectionDiffusionSine<3>;
  using BC = SineBC<3>;
  constexpr int nD = Space::nDOF, nc = 1, nSv = Space::nFaceVertices;
  const double b[3] = {1.0, 0.5, 0.25}, om[3] = {3.0, 2.0, 4.0};
  const Co coef(b, 0.2, om, 0.3);
  const WeakDirichletParams wd = WeakDirichletParams::SIPG();
  std::vector<double> qpf, qwf;
  faceQuadrature(Space::nVertices == 8, qpf, qwf);
  const int nQf = (int)qwf.size();

  // exterior faces: those that belong to one element only
  std::map<std::array<int, 4>, int> count;
  auto key = [&](int e, int k) {
    std::array<int, 4> a = {-1, -1, -1, -1};
    for (int v = 0; v < nSv; ++v) a[v] = m.conn[e * nD + Space::faceVertex(k, v)];
    std::sort(a.begin(), a.begin() + nSv);
    return a;
  };
  for (int e = 0; e < m.nE; ++e)
    for (int k = 0; k < Space::nFaces; ++k) ++count[key(e, k)];
  std::vector<int> faceElem, faceLocal;
  for (int e = 0; e < m.nE; ++e)
    for (int k = 0; k < Space::nFaces; ++k)
      if (count[key(e, k)] == 1) { faceElem.push_back(e); faceLocal.push_back(k); }
  const int nF = (int)faceElem.size();
  std::vector<int> faceKind(nF);
  for (int F = 0; F < nF; ++F) faceKind[F] = 1 + F % 4;      // flux, flux with advection, weak Dirichlet, Robin

  std::vector<double> U(m.nNodes);
  uint64_t s = 88172645463325252ULL;
  for (auto &v : U) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v = 0.2 + 0.8 * ((s >> 11) * (1.0 / 9007199254740992.0)); }
  std::vector<double> xn((size_t)nF * nD * 3), ue((size_t)nF * nD);
  for (int F = 0; F < nF; ++F)
    for (int a = 0; a < nD; ++a)
    {
      const int n = m.conn[faceElem[F] * nD + a];
      for (int d = 0; d < 3; ++d) xn[((size_t)F * nD + a) * 3 + d] = m.X[n * 3 + d];
      ue[(size_t)F * nD + a] = U[n];
    }

  // host reference
  std::vector<double> Rref((size_t)nF * nD), Jref((size_t)nF * nD * nD);
  const BC hostBC(b, 0.2, om, 0.3, faceKind.data(), 0.7, 0.4);
  const double tHost = best(repeats > 2 ? 2 : repeats, [&] {
    exteriorFaceKernel<Co, BC, Space>(coef, hostBC, wd, nF, nQf, faceLocal.data(), xn.data(), ue.data(), qpf.data(), qwf.data(), 0.0,
                                     Rref.data(), Jref.data());
  });

  // Kokkos, per face
  DView<int> dkind = toDevice("kind", faceKind), dlocal = toDevice("local", faceLocal);
  DView<double> dxn = toDevice("xn", xn), due = toDevice("ue", ue), dqp = toDevice("qpf", qpf), dqw = toDevice("qwf", qwf);
  DView<double> dR("Rf", Rref.size()), dJ("Jf", Jref.size());
  const BC devBC(b, 0.2, om, 0.3, dkind.data(), 0.7, 0.4);
  auto run1 = [&] { exteriorFaceKernelKokkos<Co, BC, Space>(coef, devBC, wd, nF, nQf, dlocal, dxn, due, dqp, dqw, 0.0, dR, dJ); };
  run1();
  const double tK = best(repeats, run1);
  std::vector<double> Rk = toHost(dR), Jk = toHost(dJ);
  const bool bitwise = Rk == Rref && Jk == Jref;

  // the colored scatter of the faces into the global system, against a serial scatter in face order
  std::vector<int> rowptr, colidx;
  ColoredMesh M = makeColoredMesh(m.nNodes, m.nE, nD, m.conn, rowptr, colidx);
  ColoredFaces CF = makeColoredFaces(M, faceElem, faceLocal);
  std::vector<int> blockOf = toHost(M.blockOf);
  std::vector<double> Rg_ref(m.nNodes, 0.0), Jv_ref(M.nBlocks, 0.0);
  for (int F = 0; F < nF; ++F)
    for (int a = 0; a < nD; ++a)
    {
      const int n = m.conn[faceElem[F] * nD + a];
      Rg_ref[n] += Rref[(size_t)F * nD + a];
      for (int bb = 0; bb < nD; ++bb) Jv_ref[blockOf[((size_t)faceElem[F] * nD + a) * nD + bb]] += Jref[((size_t)F * nD + a) * nD + bb];
    }
  DView<double> dX = toDevice("X", m.X), dU = toDevice("U", U), dRg("Rg", m.nNodes), dJv("Jv", M.nBlocks);
  auto run2 = [&] {
    Kokkos::deep_copy(dRg, 0.0); Kokkos::deep_copy(dJv, 0.0);
    assembleFacesColored<Co, BC, Space>(coef, devBC, wd, M, CF, nQf, dX, dU, dqp, dqw, 0.0, dRg, dJv);
  };
  run2();
  const double tC = best(repeats, run2);
  std::vector<double> Rg = toHost(dRg), Jv = toHost(dJv);

  std::printf("%-26s %-7s thr=%-3d faces=%-7d | face kernel %7.1f ns/face (host serial %7.1f, x%.1f) vs host: R %.1e J %.1e %s"
              " | colored faces %7.1f ns/face (%d passes) vs serial scatter: R %.1e J %.1e  checksum %016llx %016llx\n",
              name, Kokkos::DefaultExecutionSpace::name(), (int)Kokkos::DefaultExecutionSpace().concurrency(), nF, 1e9 * tK / nF,
              1e9 * tHost / nF, tHost / tK, relDiff(Rk, Rref), relDiff(Jk, Jref), bitwise ? "bitwise" : "       ", 1e9 * tC / nF,
              (int)(M.nColors * std::max(CF.nRank, 1)), relDiff(Rg, Rg_ref), relDiff(Jv, Jv_ref), (unsigned long long)fnv(Rg),
              (unsigned long long)fnv(Jv));
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
    std::vector<double> tqp, tqw, cqp, cqw;
    tetQuadrature(tqp, tqw); cubeGauss(cqp, cqw);
    const double b[3] = {1.0, 0.5, 0.25}, om[3] = {3.0, 2.0, 4.0};
    const bool spaces = !std::getenv("ADRH_BENCH_P1_ONLY");
    runCase<Poisson3D, KindP1<Poisson3D, 3>>("poisson3d (nc=1)", Poisson3D(), m, tqp, tqw, false, repeats);
    runCase<AdvectionDiffusionSine<3>, KindP1<AdvectionDiffusionSine<3>, 3>>("ad_sine (nc=1)", AdvectionDiffusionSine<3>(b, 0.2, om, 0.3), m, tqp, tqw, false, repeats);
    runCase<Coupled2, KindP1<Coupled2, 3>>("coupled2 (nc=2, mass)", Coupled2(), m, tqp, tqw, true, repeats);
    runCase<Chain5<true>, KindP1<Chain5<true>, 3>>("chain5 (nc=5, mass)", Chain5<true>(), m, tqp, tqw, true, repeats);
    if (spaces)
    {
      // the Lagrange-space kernel (engine.h): P1 through the general path, P2 tetrahedra, Q1 hexahedra
      using AD = AdvectionDiffusionSine<3>;
      const AD ad(b, 0.2, om, 0.3);
      const Tets m2 = cubeTetsP2(nn), mh = cubeHexes(nn);
      runCase<AD, KindSpace<AD, P1Simplex<3>>>("ad_sine P1 (space kernel)", ad, m, tqp, tqw, false, repeats);
      runCase<AD, KindSpace<AD, P2Simplex<3>>>("ad_sine P2 (space kernel)", ad, m2, tqp, tqw, false, repeats);
      runCase<AD, KindSpace<AD, Q1Cube<3>>>("ad_sine Q1 hex (space kernel)", ad, mh, cqp, cqw, false, repeats);
      runCase<Coupled2, KindSpace<Coupled2, P2Simplex<3>>>("coupled2 P2 (space kernel)", Coupled2(), m2, tqp, tqw, true, repeats);
      runCase<Coupled2, KindSpace<Coupled2, Q1Cube<3>>>("coupled2 Q1 hex (space kernel)", Coupled2(), mh, cqp, cqw, true, repeats);
      runFaceCase<P1Simplex<3>>("faces P1 (sine, 4 kinds)", m, repeats);
      runFaceCase<P2Simplex<3>>("faces P2 (sine, 4 kinds)", m2, repeats);
      runFaceCase<Q1Cube<3>>("faces Q1 hex (sine, 4 kinds)", mh, repeats);
    }
  }
  Kokkos::finalize();
  return 0;
}

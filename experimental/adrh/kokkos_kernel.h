// The ADRH element kernel as Kokkos kernels (research port, see ADRH_ENGINE_DESIGN.md section 19).
//
// The per-element function is adrh::elementMaskedOne (masked.h); here it is run by Kokkos::parallel_for on
// whatever execution space Kokkos was built for (Serial, OpenMP, CUDA, ...). Nothing in the element function
// changed: the same source is the host prototype's and the device kernel's.
//
//   elementKernelKokkos   one thread per element, R and J of every element stored (the prototype's interface).
//   assembleColored       fused assembly: a thread computes its element and adds it to the global residual
//                         vector and the global block-CSR Jacobian. Elements of one color share no node, so
//                         the colors run one after the other and each color's scatter needs no atomics or
//                         locks: the result is the same bits at every thread count and on every backend that
//                         does the same arithmetic. (Build the coloring with colorElements.)
#pragma once
#ifndef ADRH_USE_KOKKOS
#define ADRH_USE_KOKKOS
#endif
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "coefficients.h"
#include "engine.h"

namespace adrh
{
  template <class T> using DView = Kokkos::View<T *>;

  // Run f(e) for e in [begin, end), one Kokkos iteration per element. f is a functor class, not a lambda (nvcc restricts
  // lambdas inside templates). The performance of this launch depends on the kernel being fully inlined into the
  // functor: see ADRH_LAMBDA_INLINE in portable.h.
  template <class F>
  void forEachElement(const char *label, int begin, int end, const F &f)
  {
    Kokkos::parallel_for(label, Kokkos::RangePolicy<>(begin, end), f);
  }

  // What one element is: its node count and the function that computes it. P1Element is the affine simplex kernel
  // (masked.h); SpaceElement is the kernel for any Lagrange space (engine.h, spaces.h: P1, P2, Q1; isoparametric geometry).
  template <class Coefficients, int nS_>
  struct P1Element
  {
    static constexpr int nD = nS_ + 1;
    KOKKOS_INLINE_FUNCTION static void compute(const Coefficients &c, const double *X, const double *ue, const double *uo,
                                               int nQ, const double *qp, const double *qw, double t, double dt, double *R,
                                               double *J)
    {
      elementMaskedOne<Coefficients, nS_>(c, X, ue, uo, nQ, qp, qw, t, dt, R, J);
    }
  };
  template <class Coefficients, class Space>
  struct SpaceElement
  {
    static constexpr int nD = Space::nDOF;
    KOKKOS_INLINE_FUNCTION static void compute(const Coefficients &c, const double *X, const double *ue, const double *uo,
                                               int nQ, const double *qp, const double *qw, double t, double dt, double *R,
                                               double *J)
    {
      elementSpaceOne<Coefficients, Space>(c, X, ue, uo, nQ, qp, qw, t, dt, R, J);
    }
  };

  // Flat views in the prototype's layouts: xn (nE, nD, 3), u and uold (nE, nc, nD), R (nE, nc, nD),
  // J (nE, nc, nD, nc, nD), nD the element's DOF count. uold has extent 0 for a steady problem.
  //
  // Steady against unsteady is a compile-time choice (the Old parameter), so that a null old-time pointer is
  // visible to the compiler and the old-time work disappears.
  template <class Coefficients, class Element, bool Old>
  struct ElementFunctor
  {
    static constexpr int nc = Coefficients::nc, nD = Element::nD;
    Coefficients c;
    int nQ;
    double t, dt;
    DView<const double> xn, u, uold, qp, qw;
    DView<double> R, J;
    KOKKOS_INLINE_FUNCTION void operator()(const int e) const
    {
      Element::compute(c, xn.data() + (size_t)e * nD * 3, u.data() + (size_t)e * nc * nD,
                       Old ? uold.data() + (size_t)e * nc * nD : nullptr, nQ, qp.data(), qw.data(), t, dt,
                       R.data() + (size_t)e * nc * nD, J.data() + (size_t)e * nc * nD * nc * nD);
    }
  };

  template <class Coefficients, class Element>
  void elementKernelLaunch(const Coefficients &coef, int nE, int nQ, DView<const double> xn, DView<const double> u,
                           DView<const double> uold, DView<const double> qp, DView<const double> qw, double t, double dt,
                           DView<double> R, DView<double> J)
  {
    if (uold.extent(0) > 0)
      forEachElement("adrh element kernel", 0, nE,
                     ElementFunctor<Coefficients, Element, true>{coef, nQ, t, dt, xn, u, uold, qp, qw, R, J});
    else
      forEachElement("adrh element kernel", 0, nE,
                     ElementFunctor<Coefficients, Element, false>{coef, nQ, t, dt, xn, u, uold, qp, qw, R, J});
    Kokkos::fence();
  }

  // the P1 simplex kernel (the prototype's interface)
  template <class Coefficients, int nS>
  void elementKernelKokkos(const Coefficients &coef, int nE, int nQ, DView<const double> xn, DView<const double> u,
                           DView<const double> uold, DView<const double> qp, DView<const double> qw, double t, double dt,
                           DView<double> R, DView<double> J)
  {
    elementKernelLaunch<Coefficients, P1Element<Coefficients, nS>>(coef, nE, nQ, xn, u, uold, qp, qw, t, dt, R, J);
  }

  // any Lagrange space (P1, P2 simplices; Q1 cubes)
  template <class Coefficients, class Space>
  void elementKernelKokkosSpace(const Coefficients &coef, int nE, int nQ, DView<const double> xn, DView<const double> u,
                                DView<const double> uold, DView<const double> qp, DView<const double> qw, double t,
                                double dt, DView<double> R, DView<double> J)
  {
    elementKernelLaunch<Coefficients, SpaceElement<Coefficients, Space>>(coef, nE, nQ, xn, u, uold, qp, qw, t, dt, R, J);
  }

  // ---- fused, colored assembly -------------------------------------------------------------------------

  // Connectivity, the elements grouped by color, and the position of each element's (a, b) node pair in the
  // block-CSR Jacobian (a block is nc x nc, rows and columns are nodes).
  struct ColoredMesh
  {
    int nNodes = 0, nE = 0, nD = 0, nColors = 0;
    DView<int> conn;                        // (nE, nD) node numbers
    DView<int> order;                       // element numbers, grouped by color
    std::vector<int> colorBegin;            // host: color c is order[colorBegin[c] .. colorBegin[c+1])
    DView<int> blockOf;                     // (nE, nD, nD) index of block (node a, node b) in the CSR values
    int nBlocks = 0;
  };

  // Greedy coloring: an element gets the lowest color no neighbouring (node-sharing) element has. Returns the
  // color of every element. Deterministic (elements in numbering order).
  inline std::vector<int> colorElements(int nNodes, int nE, int nD, const int *conn, int &nColors)
  {
    std::vector<uint64_t> used(nNodes, 0);
    std::vector<int> color(nE);
    nColors = 0;
    for (int e = 0; e < nE; ++e)
    {
      uint64_t m = 0;
      for (int a = 0; a < nD; ++a) m |= used[conn[e * nD + a]];
      int c = 0;
      while (m >> c & 1) ++c;
      if (c >= 64) throw std::runtime_error("more than 64 element colors");
      color[e] = c;
      nColors = c + 1 > nColors ? c + 1 : nColors;
      for (int a = 0; a < nD; ++a) used[conn[e * nD + a]] |= uint64_t(1) << c;
    }
    return color;
  }

  // Block-CSR pattern of the nodal graph (rowptr, colidx on the host) and, per element and node pair, the
  // index of the block. Rows hold sorted columns, so the lookup is a binary search.
  inline void buildBlockCsr(int nNodes, int nE, int nD, const int *conn, std::vector<int> &rowptr,
                            std::vector<int> &colidx, std::vector<int> &blockOf)
  {
    std::vector<std::vector<int>> adj(nNodes);
    for (int e = 0; e < nE; ++e)
      for (int a = 0; a < nD; ++a)
        for (int b = 0; b < nD; ++b) adj[conn[e * nD + a]].push_back(conn[e * nD + b]);
    rowptr.assign(nNodes + 1, 0);
    colidx.clear();
    for (int n = 0; n < nNodes; ++n)
    {
      std::sort(adj[n].begin(), adj[n].end());
      adj[n].erase(std::unique(adj[n].begin(), adj[n].end()), adj[n].end());
      colidx.insert(colidx.end(), adj[n].begin(), adj[n].end());
      rowptr[n + 1] = (int)colidx.size();
    }
    blockOf.resize((size_t)nE * nD * nD);
    for (int e = 0; e < nE; ++e)
      for (int a = 0; a < nD; ++a)
        for (int b = 0; b < nD; ++b)
        {
          const int row = conn[e * nD + a], col = conn[e * nD + b];
          const int *lo = colidx.data() + rowptr[row], *hi = colidx.data() + rowptr[row + 1];
          blockOf[((size_t)e * nD + a) * nD + b] = (int)(std::lower_bound(lo, hi, col) - colidx.data());
        }
  }

  inline ColoredMesh makeColoredMesh(int nNodes, int nE, int nD, const std::vector<int> &conn,
                                     std::vector<int> &rowptr, std::vector<int> &colidx)
  {
    ColoredMesh M;
    M.nNodes = nNodes; M.nE = nE; M.nD = nD;
    std::vector<int> color = colorElements(nNodes, nE, nD, conn.data(), M.nColors);
    M.colorBegin.assign(M.nColors + 1, 0);
    for (int e = 0; e < nE; ++e) ++M.colorBegin[color[e] + 1];
    for (int c = 0; c < M.nColors; ++c) M.colorBegin[c + 1] += M.colorBegin[c];
    std::vector<int> order(nE), fill(M.colorBegin.begin(), M.colorBegin.end() - 1);
    for (int e = 0; e < nE; ++e) order[fill[color[e]]++] = e;
    std::vector<int> blockOf;
    buildBlockCsr(nNodes, nE, nD, conn.data(), rowptr, colidx, blockOf);
    M.nBlocks = (int)colidx.size();
    auto upload = [](const char *name, const std::vector<int> &h) {
      DView<int> d(name, h.size());
      auto m = Kokkos::create_mirror_view(d);
      for (size_t k = 0; k < h.size(); ++k) m(k) = h[k];
      Kokkos::deep_copy(d, m);
      return d;
    };
    M.conn = upload("conn", conn);
    M.order = upload("order", order);
    M.blockOf = upload("blockOf", blockOf);
    return M;
  }

  // One color's elements share no node, so the scatter needs neither atomics nor locks.
  template <class Coefficients, class Element, bool Old>
  struct ColorFunctor
  {
    static constexpr int nc = Coefficients::nc, nD = Element::nD;
    Coefficients cf;
    int nQ;
    double t, dt;
    DView<const int> conn, order, blockOf;
    DView<const double> X, U, Uold, qp, qw;
    DView<double> Rg, Jv;
    KOKKOS_INLINE_FUNCTION void operator()(const int k) const
    {
      const int e = order(k);
      double Xe[nD * 3], ue[nc * nD], uo[nc * nD], Re[nc * nD], Je[nc * nD * nc * nD];
      for (int a = 0; a < nD; ++a)
      {
        const int n = conn(e * nD + a);
        for (int I = 0; I < 3; ++I) Xe[a * 3 + I] = X(n * 3 + I);
        for (int j = 0; j < nc; ++j)
        {
          ue[j * nD + a] = U(n * nc + j);
          if (Old) uo[j * nD + a] = Uold(n * nc + j);
        }
      }
      Element::compute(cf, Xe, ue, Old ? uo : nullptr, nQ, qp.data(), qw.data(), t, dt, Re, Je);
      for (int a = 0; a < nD; ++a)
      {
        const int n = conn(e * nD + a);
        for (int i = 0; i < nc; ++i) Rg(n * nc + i) += Re[i * nD + a];
        for (int b = 0; b < nD; ++b)
        {
          double *blk = Jv.data() + (size_t)blockOf(((size_t)e * nD + a) * nD + b) * nc * nc;
          for (int i = 0; i < nc; ++i)
            for (int j = 0; j < nc; ++j) blk[i * nc + j] += Je[((i * nD + a) * nc + j) * nD + b];
        }
      }
    }
  };

  template <class Coefficients, class Element, bool Old>
  void assembleColor(const Coefficients &coef, const ColoredMesh &M, int c, int nQ, DView<const double> X,
                     DView<const double> U, DView<const double> Uold, DView<const double> qp,
                     DView<const double> qw, double t, double dt, DView<double> Rg, DView<double> Jv)
  {
    forEachElement("adrh colored assembly", M.colorBegin[c], M.colorBegin[c + 1],
                   ColorFunctor<Coefficients, Element, Old>{coef, nQ, t, dt, M.conn, M.order, M.blockOf, X, U, Uold, qp, qw,
                                                       Rg, Jv});
  }

  // Zero the global residual (nNodes * nc) and block values (nBlocks * nc * nc), then one launch per color.
  // X (nNodes, 3) coordinates of every DOF, U and Uold (nNodes, nc) nodal values (Uold extent 0 for steady). M must have been
  // made with the element's DOF count.
  template <class Coefficients, class Element>
  void assembleColoredElement(const Coefficients &coef, const ColoredMesh &M, int nQ, DView<const double> X,
                              DView<const double> U, DView<const double> Uold, DView<const double> qp,
                              DView<const double> qw, double t, double dt, DView<double> Rg, DView<double> Jv)
  {
    Kokkos::deep_copy(Rg, 0.0);
    Kokkos::deep_copy(Jv, 0.0);
    const bool old = Uold.extent(0) > 0;
    for (int c = 0; c < M.nColors; ++c)
      if (old)
        assembleColor<Coefficients, Element, true>(coef, M, c, nQ, X, U, Uold, qp, qw, t, dt, Rg, Jv);
      else
        assembleColor<Coefficients, Element, false>(coef, M, c, nQ, X, U, Uold, qp, qw, t, dt, Rg, Jv);
    Kokkos::fence();
  }

  // the P1 simplex kernel
  template <class Coefficients, int nS>
  void assembleColored(const Coefficients &coef, const ColoredMesh &M, int nQ, DView<const double> X, DView<const double> U,
                       DView<const double> Uold, DView<const double> qp, DView<const double> qw, double t, double dt,
                       DView<double> Rg, DView<double> Jv)
  {
    assembleColoredElement<Coefficients, P1Element<Coefficients, nS>>(coef, M, nQ, X, U, Uold, qp, qw, t, dt, Rg, Jv);
  }

  // any Lagrange space
  template <class Coefficients, class Space>
  void assembleColoredSpace(const Coefficients &coef, const ColoredMesh &M, int nQ, DView<const double> X,
                            DView<const double> U, DView<const double> Uold, DView<const double> qp, DView<const double> qw,
                            double t, double dt, DView<double> Rg, DView<double> Jv)
  {
    assembleColoredElement<Coefficients, SpaceElement<Coefficients, Space>>(coef, M, nQ, X, U, Uold, qp, qw, t, dt, Rg, Jv);
  }
} // namespace adrh

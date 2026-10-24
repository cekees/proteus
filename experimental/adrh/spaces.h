// Lagrange spaces on a reference element, and the isoparametric geometry built from them.
//
// A Space gives, at a reference point xi (nS coordinates): the basis values w[nDOF] and reference gradients
// gwref[nDOF][nS]; the map from a face's own coordinates eta (nS - 1 of them, the domain of the face
// quadrature) to xi; the outward reference normal of each face, scaled so that
//     n dS = |det J| J^{-T} n_ref d(eta)
// holds against the face quadrature weights (so the slanted simplex face, parametrized by the standard
// reference simplex, gets n_ref = (1, ..., 1) with |n_ref| = sqrt(nS) = the stretch of that parametrization);
// and the vertices of each face (the penalty's h is the longest edge between them).
//
// Geometry is isoparametric through the same basis: J[I][K] = sum_b X_b[I] gwref[b][K]. For simplices with the
// P2 nodes at the edge midpoints this is the affine map, exactly; for hexahedra it is the trilinear map.
// The vertices are always the first nVertices DOFs, and the element's DOF coordinates are what the kernels get
// (nDOF points with 3 coordinates each, as proteus stores them).
#pragma once
#include <cmath>

namespace adrh
{
  template <int nS_>
  struct SimplexBase
  {
    static constexpr int nS = nS_, nVertices = nS_ + 1, nFaces = nS_ + 1, nFaceVertices = nS_;
    static_assert(nS_ == 2 || nS_ == 3, "triangles or tetrahedra");
    // barycentric coordinates lambda_0 = 1 - sum xi, lambda_i = xi_{i-1}, and their constant gradients
    static void bary(const double *xi, double *lam, double (*glam)[nS_])
    {
      lam[0] = 1.0;
      for (int I = 0; I < nS_; ++I) { lam[0] -= xi[I]; glam[0][I] = -1.0; }
      for (int i = 1; i <= nS_; ++i)
      {
        lam[i] = xi[i - 1];
        for (int I = 0; I < nS_; ++I) glam[i][I] = (I == i - 1) ? 1.0 : 0.0;
      }
    }
    // face k is opposite vertex k; its vertices in increasing order
    static int faceVertex(int k, int m) { return m < k ? m : m + 1; }
    // eta (nS - 1) on the reference (nS-1)-simplex -> xi on face k: lambda_k = 0, lambda_{faceVertex(k,0)} = 1 - sum eta,
    // lambda_{faceVertex(k,m)} = eta_{m-1}
    static void faceToReference(int k, const double *eta, double *xi)
    {
      double lam[nS_ + 1];
      lam[k] = 0.0;
      lam[faceVertex(k, 0)] = 1.0;
      for (int m = 1; m < nS_; ++m)
      {
        lam[faceVertex(k, m)] = eta[m - 1];
        lam[faceVertex(k, 0)] -= eta[m - 1];
      }
      for (int i = 1; i <= nS_; ++i) xi[i - 1] = lam[i];
    }
    static void referenceNormal(int k, double *n)
    {
      for (int I = 0; I < nS_; ++I) n[I] = (k == 0) ? 1.0 : ((I == k - 1) ? -1.0 : 0.0);
    }
  };

  template <int nS_>
  struct P1Simplex : SimplexBase<nS_>
  {
    static constexpr int nDOF = nS_ + 1;
    static void basis(const double *xi, double *w, double (*gw)[nS_]) { SimplexBase<nS_>::bary(xi, w, gw); }
  };

  // Quadratic Lagrange on simplices, proteus's numbering (QuadraticOnSimplexWithNodalBasis): vertices, then the
  // edge midpoints in the order 2D (0,1), (1,2), (0,2); 3D (0,1), (1,2), (2,3), (0,2), (1,3), (0,3).
  template <int nS_>
  struct P2Simplex : SimplexBase<nS_>
  {
    static constexpr int nDOF = (nS_ + 1) * (nS_ + 2) / 2, nEdges = nDOF - nS_ - 1;
    static constexpr int edges2[3][2] = {{0, 1}, {1, 2}, {0, 2}};
    static constexpr int edges3[6][2] = {{0, 1}, {1, 2}, {2, 3}, {0, 2}, {1, 3}, {0, 3}};
    static constexpr int edge(int e, int end) { return nS_ == 2 ? edges2[e][end] : edges3[e][end]; }
    static void basis(const double *xi, double *w, double (*gw)[nS_])
    {
      double lam[nS_ + 1], glam[nS_ + 1][nS_];
      SimplexBase<nS_>::bary(xi, lam, glam);
      for (int i = 0; i <= nS_; ++i)
      {
        w[i] = lam[i] * (2.0 * lam[i] - 1.0);
        for (int I = 0; I < nS_; ++I) gw[i][I] = (4.0 * lam[i] - 1.0) * glam[i][I];
      }
      for (int e = 0; e < nEdges; ++e)
      {
        const int j = edge(e, 0), k = edge(e, 1), d = nS_ + 1 + e;
        w[d] = 4.0 * lam[j] * lam[k];
        for (int I = 0; I < nS_; ++I) gw[d][I] = 4.0 * lam[k] * glam[j][I] + 4.0 * lam[j] * glam[k][I];
      }
    }
  };

  // The geometry at one reference point: basis values w, physical gradients gw, the point x, |det J| and J^{-1}.
  template <class Space>
  struct PointGeometry
  {
    static constexpr int nS = Space::nS, nDOF = Space::nDOF;
    double w[nDOF], gw[nDOF][nS], x[3], Jinv[nS][nS], adet;
    // X: the element's nDOF node coordinates, 3 each
    void compute(const double *X, const double *xi)
    {
      double gwref[nDOF][nS];
      Space::basis(xi, w, gwref);
      double Jm[nS][nS] = {};
      for (int b = 0; b < nDOF; ++b)
        for (int I = 0; I < nS; ++I)
          for (int K = 0; K < nS; ++K) Jm[I][K] += X[b * 3 + I] * gwref[b][K];
      double det;
      if constexpr (nS == 2)
      {
        det = Jm[0][0] * Jm[1][1] - Jm[0][1] * Jm[1][0];
        Jinv[0][0] = Jm[1][1] / det; Jinv[0][1] = -Jm[0][1] / det;
        Jinv[1][0] = -Jm[1][0] / det; Jinv[1][1] = Jm[0][0] / det;
      }
      else
      {
        det = Jm[0][0] * (Jm[1][1] * Jm[2][2] - Jm[1][2] * Jm[2][1])
            - Jm[0][1] * (Jm[1][0] * Jm[2][2] - Jm[1][2] * Jm[2][0])
            + Jm[0][2] * (Jm[1][0] * Jm[2][1] - Jm[1][1] * Jm[2][0]);
        Jinv[0][0] = (Jm[1][1] * Jm[2][2] - Jm[1][2] * Jm[2][1]) / det;
        Jinv[0][1] = (Jm[0][2] * Jm[2][1] - Jm[0][1] * Jm[2][2]) / det;
        Jinv[0][2] = (Jm[0][1] * Jm[1][2] - Jm[0][2] * Jm[1][1]) / det;
        Jinv[1][0] = (Jm[1][2] * Jm[2][0] - Jm[1][0] * Jm[2][2]) / det;
        Jinv[1][1] = (Jm[0][0] * Jm[2][2] - Jm[0][2] * Jm[2][0]) / det;
        Jinv[1][2] = (Jm[0][2] * Jm[1][0] - Jm[0][0] * Jm[1][2]) / det;
        Jinv[2][0] = (Jm[1][0] * Jm[2][1] - Jm[1][1] * Jm[2][0]) / det;
        Jinv[2][1] = (Jm[0][1] * Jm[2][0] - Jm[0][0] * Jm[2][1]) / det;
        Jinv[2][2] = (Jm[0][0] * Jm[1][1] - Jm[0][1] * Jm[1][0]) / det;
      }
      adet = std::fabs(det);
      for (int b = 0; b < nDOF; ++b)
        for (int I = 0; I < nS; ++I)
        {
          gw[b][I] = 0.0;
          for (int K = 0; K < nS; ++K) gw[b][I] += gwref[b][K] * Jinv[K][I];
        }
      for (int I = 0; I < 3; ++I)
      {
        x[I] = 0.0;
        for (int b = 0; b < nDOF; ++b) x[I] += w[b] * X[b * 3 + I];
      }
    }
    // outward unit normal of face k at this point and the face measure per unit face-quadrature weight
    void faceNormal(int k, double *nrm, double &vn) const
    {
      double nref[nS];
      Space::referenceNormal(k, nref);
      double v[3] = {0.0, 0.0, 0.0};
      for (int I = 0; I < nS; ++I)
        for (int K = 0; K < nS; ++K) v[I] += adet * Jinv[K][I] * nref[K];
      vn = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      for (int I = 0; I < 3; ++I) nrm[I] = v[I] / vn;
    }
  };
} // namespace adrh

namespace adrh
{
  // Trilinear (bilinear) Lagrange on proteus's reference cube [-1, 1]^nS, nodes in proteus's order
  // (ReferenceCube.nodeList): 2D (-,-), (+,-), (+,+), (-,+); 3D the same square at xi_2 = -1 then at +1.
  // Faces are numbered k = 2 I + side: xi_I = -1 (side 0) or +1 (side 1); a face's own coordinates are the other
  // reference coordinates in increasing order, on [-1, 1]^(nS-1) like proteus's CubeGaussQuadrature(nS - 1).
  template <int nS_>
  struct Q1Cube
  {
    static constexpr int nS = nS_, nDOF = 1 << nS_, nVertices = nDOF, nFaces = 2 * nS_, nFaceVertices = 1 << (nS_ - 1);
    static_assert(nS_ == 2 || nS_ == 3, "quadrilaterals or hexahedra");
    // the sign of node b's reference coordinate I
    static constexpr double sgn(int b, int I)
    {
      const int q = b % 4;
      if (I == 0) return (q == 1 || q == 2) ? 1.0 : -1.0;
      if (I == 1) return (q >= 2) ? 1.0 : -1.0;
      return (b >= 4) ? 1.0 : -1.0;
    }
    static void basis(const double *xi, double *w, double (*gw)[nS_])
    {
      constexpr double scale = 1.0 / nDOF;
      for (int b = 0; b < nDOF; ++b)
      {
        double f[nS_];
        for (int I = 0; I < nS_; ++I) f[I] = 1.0 + sgn(b, I) * xi[I];
        w[b] = scale;
        for (int I = 0; I < nS_; ++I) w[b] *= f[I];
        for (int I = 0; I < nS_; ++I)
        {
          gw[b][I] = scale * sgn(b, I);
          for (int J = 0; J < nS_; ++J)
            if (J != I) gw[b][I] *= f[J];
        }
      }
    }
    static int faceVertex(int k, int m)
    {
      const int I = k / 2;
      const double side = (k % 2) ? 1.0 : -1.0;
      int seen = 0;
      for (int b = 0; b < nDOF; ++b)
        if (sgn(b, I) == side && seen++ == m) return b;
      return -1;
    }
    static void faceToReference(int k, const double *eta, double *xi)
    {
      const int I = k / 2;
      int m = 0;
      for (int J = 0; J < nS_; ++J) xi[J] = (J == I) ? ((k % 2) ? 1.0 : -1.0) : eta[m++];
    }
    static void referenceNormal(int k, double *n)
    {
      for (int J = 0; J < nS_; ++J) n[J] = (J == k / 2) ? ((k % 2) ? 1.0 : -1.0) : 0.0;
    }
  };
} // namespace adrh

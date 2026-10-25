// Prototype of the n-component ADRH element kernel with an inline coefficient functor.
//
// Not part of the proteus build. It exists to settle the interface a ymf code generator
// has to target (see ADRH_ENGINE_DESIGN.md): the coefficient functor is a template
// parameter, evaluated at each quadrature point inside the element loop; nothing is stored
// per quadrature point; geometry is computed in the kernel. P1 simplices only, Galerkin only.
#pragma once
#include <cmath>
#include <cstring>
#include "portable.h"

namespace adrh
{
  // Proteus's form, per component i:  m^i_t + div(f^i - sum_k a^{ik} grad phi^k) + H^i + r^i = 0
  template <int nc, int nS>
  struct Terms
  {
    double m[nc], dm[nc][nc];                           // mass, d/du_j
    double f[nc][nS], df[nc][nc][nS];                   // advection, d/du_j
    double phi[nc], dphi[nc][nc];                       // diffusion potentials (phi = u for plain diffusion)
    double a[nc][nc][nS][nS], da[nc][nc][nc][nS][nS];   // a^{ik}, d a^{ik}/du_j
    double H[nc], dH[nc][nc][nS];                       // Hamiltonian, d/d(grad u_j)
    double r[nc], dr[nc][nc];                           // reaction, d/du_j
    ADRH_FN void zero()
    {
      double *p = reinterpret_cast<double *>(this);   // all members are doubles
      for (unsigned k = 0; k < sizeof(*this) / sizeof(double); ++k) p[k] = 0.0;
    }
  };

  // Geometry of an affine P1 simplex: physical gradients of the basis (constant on the element) and |det J|.
  template <int nS>
  ADRH_FN void affineP1(const double *X, double gw[nS + 1][nS], double &adet)
  {
    constexpr int nD = nS + 1;
    double Jm[nS][nS], Ji[nS][nS], det = 0.0;
    for (int I = 0; I < nS; ++I)
      for (int K = 0; K < nS; ++K)
        Jm[I][K] = X[(K + 1) * 3 + I] - X[0 * 3 + I];
    if constexpr (nS == 2)
    {
      det = Jm[0][0] * Jm[1][1] - Jm[0][1] * Jm[1][0];
      Ji[0][0] = Jm[1][1] / det; Ji[0][1] = -Jm[0][1] / det;
      Ji[1][0] = -Jm[1][0] / det; Ji[1][1] = Jm[0][0] / det;
    }
    else
    {
      det = Jm[0][0] * (Jm[1][1] * Jm[2][2] - Jm[1][2] * Jm[2][1])
          - Jm[0][1] * (Jm[1][0] * Jm[2][2] - Jm[1][2] * Jm[2][0])
          + Jm[0][2] * (Jm[1][0] * Jm[2][1] - Jm[1][1] * Jm[2][0]);
      Ji[0][0] = (Jm[1][1] * Jm[2][2] - Jm[1][2] * Jm[2][1]) / det;
      Ji[0][1] = (Jm[0][2] * Jm[2][1] - Jm[0][1] * Jm[2][2]) / det;
      Ji[0][2] = (Jm[0][1] * Jm[1][2] - Jm[0][2] * Jm[1][1]) / det;
      Ji[1][0] = (Jm[1][2] * Jm[2][0] - Jm[1][0] * Jm[2][2]) / det;
      Ji[1][1] = (Jm[0][0] * Jm[2][2] - Jm[0][2] * Jm[2][0]) / det;
      Ji[1][2] = (Jm[0][2] * Jm[1][0] - Jm[0][0] * Jm[1][2]) / det;
      Ji[2][0] = (Jm[1][0] * Jm[2][1] - Jm[1][1] * Jm[2][0]) / det;
      Ji[2][1] = (Jm[0][1] * Jm[2][0] - Jm[0][0] * Jm[2][1]) / det;
      Ji[2][2] = (Jm[0][0] * Jm[1][1] - Jm[0][1] * Jm[1][0]) / det;
    }
    adet = std::fabs(det);
    for (int I = 0; I < nS; ++I)
      gw[0][I] = 0.0;
    for (int b = 1; b < nD; ++b)
      for (int I = 0; I < nS; ++I)
      {
        gw[b][I] = Ji[b - 1][I];
        gw[0][I] -= Ji[b - 1][I];
      }
  }

  // Element residual and Jacobian for P1 on simplices (nS+1 nodes), all elements.
  //   xn  (nE, nS+1, 3)  node coordinates       u, uold (nE, nc, nS+1)  element DOF values
  //   qp  (nQ, nS), qw (nQ)  reference quadrature (weights sum to the reference volume)
  //   R   (nE, nc, nS+1)     J  (nE, nc, nS+1, nc, nS+1)
  // dt <= 0 means steady (no mass term and no old-time evaluation).
  template <class Coefficients, int nS>
  void elementKernel(const Coefficients &C, int nE, int nQ,
                     const double *xn, const double *u, const double *uold,
                     const double *qp, const double *qw, double t, double dt,
                     double *R, double *J)
  {
    constexpr int nc = Coefficients::nc;
    constexpr int nD = nS + 1;
    static_assert(Coefficients::nSpace == nS, "coefficient space dimension");
    for (int eN = 0; eN < nE; ++eN)
    {
      const double *X = xn + eN * nD * 3;
      double gw[nD][nS], adet;
      affineP1<nS>(X, gw, adet);
      double Rloc[nc][nD] = {};
      double Jloc[nc][nD][nc][nD] = {};
      const double *ue = u + eN * nc * nD;
      const double *uo = uold ? uold + eN * nc * nD : nullptr;
      // gradients of u_j (constant for P1)
      double gu[nc][nS], guo[nc][nS];
      for (int j = 0; j < nc; ++j)
        for (int I = 0; I < nS; ++I)
        {
          gu[j][I] = 0.0; guo[j][I] = 0.0;
          for (int b = 0; b < nD; ++b)
          {
            gu[j][I] += ue[j * nD + b] * gw[b][I];
            if (uo) guo[j][I] += uo[j * nD + b] * gw[b][I];
          }
        }
      for (int q = 0; q < nQ; ++q)
      {
        double w[nD];
        w[0] = 1.0;
        for (int I = 0; I < nS; ++I)
          w[0] -= qp[q * nS + I];
        for (int b = 1; b < nD; ++b)
          w[b] = qp[q * nS + b - 1];
        double x[3] = {0.0, 0.0, 0.0};
        for (int b = 0; b < nD; ++b)
          for (int I = 0; I < 3; ++I)
            x[I] += w[b] * X[b * 3 + I];
        const double dV = qw[q] * adet;
        double uq[nc], uoq[nc];
        for (int j = 0; j < nc; ++j)
        {
          uq[j] = 0.0; uoq[j] = 0.0;
          for (int b = 0; b < nD; ++b)
          {
            uq[j] += ue[j * nD + b] * w[b];
            if (uo) uoq[j] += uo[j * nD + b] * w[b];
          }
        }
        Terms<nc, nS> T;
        T.zero();
        C.evaluate(t, x, uq, gu, T);
        double mold[nc] = {};
        if (dt > 0.0 && uo)
        {
          Terms<nc, nS> To;
          To.zero();
          C.evaluate(t - dt, x, uoq, guo, To);
          for (int i = 0; i < nc; ++i) mold[i] = To.m[i];
        }
        const double idt = (dt > 0.0) ? 1.0 / dt : 0.0;
        // gradient of the potentials: grad phi^k = sum_j dphi[k][j] grad u_j
        double gphi[nc][nS];
        for (int k = 0; k < nc; ++k)
          for (int I = 0; I < nS; ++I)
          {
            gphi[k][I] = 0.0;
            for (int j = 0; j < nc; ++j)
              gphi[k][I] += T.dphi[k][j] * gu[j][I];
          }
        for (int i = 0; i < nc; ++i)
        {
          // flux seen by the test function gradient: -f + sum_k a^{ik} grad phi^k
          double flux[nS];
          for (int I = 0; I < nS; ++I)
          {
            flux[I] = -T.f[i][I];
            for (int k = 0; k < nc; ++k)
              for (int J = 0; J < nS; ++J)
                flux[I] += T.a[i][k][I][J] * gphi[k][J];
          }
          for (int a = 0; a < nD; ++a)
          {
            double s = (T.m[i] - mold[i]) * idt * w[a] + (T.H[i] + T.r[i]) * w[a];
            for (int I = 0; I < nS; ++I)
              s += flux[I] * gw[a][I];
            Rloc[i][a] += dV * s;
            for (int j = 0; j < nc; ++j)
              for (int b = 0; b < nD; ++b)
              {
                double jab = T.dm[i][j] * idt * w[b] * w[a] + T.dr[i][j] * w[b] * w[a];
                for (int I = 0; I < nS; ++I)
                {
                  jab += (T.dH[i][j][I] * gw[b][I]) * w[a];
                  jab -= T.df[i][j][I] * w[b] * gw[a][I];
                  for (int k = 0; k < nc; ++k)
                    for (int Jx = 0; Jx < nS; ++Jx)
                    {
                      jab += T.da[i][k][j][I][Jx] * w[b] * gphi[k][Jx] * gw[a][I];
                      jab += T.a[i][k][I][Jx] * T.dphi[k][j] * gw[b][Jx] * gw[a][I];
                    }
                }
                Jloc[i][a][j][b] += dV * jab;
              }
          }
        }
      }
      std::memcpy(R + eN * nc * nD, Rloc, sizeof(Rloc));
      std::memcpy(J + eN * nc * nD * nc * nD, Jloc, sizeof(Jloc));
    }
  }
} // namespace adrh

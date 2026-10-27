// Exterior-face terms of the weak form for P1 simplices (segments of triangles, triangles of tetrahedra),
// as a separate kernel.
//
// The Galerkin form integrated by parts leaves, on every exterior face,
//     + integral over the face of  F . n  w  dS,   F = f(u) - sum_k a^{ik} grad phi^k   (the total flux),
// which proteus adds to the element residual as (advectiveFlux + diffusiveFlux) w dS
// (femIntegrals.c updateExteriorElementBoundaryFlux). What stands in for F . n at a boundary quadrature
// point is decided per component by the BC functor, one of the kinds below. The structure follows
// proteus's exterior numerical flux classes (NumericalFlux.py, numericalFlux.c) so the two can be
// compared term by term:
//
//   Strong         nothing here; the driver removes the unknown or replaces its row
//                  (StrongDirichletFactory, or numericalFluxType=None).
//   Flux           the total flux is given. Diffusive part: g = -a grad u . n, prescribed (a
//                  diffusiveFluxBoundaryCondition), optionally affine in the trace, g0 + dg u (RobinFactory:
//                  alpha u + beta q.n = g gives q.n = g/beta - (alpha/beta) u, with the Jacobian -alpha/beta).
//                  Advective part: prescribed where a value is given (an advectiveFluxBoundaryCondition; no
//                  Jacobian, as proteus zeroes dadvectiveFlux_left there); where none is, the interior flux
//                  f(u) . n on outflow (df . n >= 0) and ZERO on inflow, which is what StrongDirichlet's
//                  'outFlow' does (calculateExteriorNumericalAdvectiveFlux_NoBC drops the inflow flux).
//   WeakDirichlet  the value g_D is imposed weakly, as Advection_DiagonalUpwind_Diffusion_{IIPG,NIPG,SIPG}_exterior
//                  do. Advective part: upwinded between f(u) . n and f(g_D) . n by the signs of df(u) . n and
//                  df(g_D) . n (calculateExteriorNumericalAdvectiveFlux: a shock speed decides the mixed case, a
//                  transonic rarefaction is averaged), with a Jacobian only where the interior flux is taken.
//                  Diffusive part: -a grad phi . n + (C / h^p) (u - g_D), h the face diameter (its longest
//                  edge; the edge itself in 2D); C, p = 2, 1 for IIPG, 10, 1 for NIPG, 100, 1 for SIPG. Adjoint part
//                  (sigma = 0 IIPG, -1 NIPG, +1 SIPG): - sigma (u - g_D) (a^T n) . grad w dS, residual and Jacobian.
//
// nc > 1: the kind is per component i. The diffusive part of component i sums over the potentials k with
// a^{ik} nonzero; penalty and adjoint are on the jump of u_k against g_D[k], and zero where component k is
// not itself WeakDirichlet (proteus: bc_u is u where no Dirichlet value is set). The upwind decision uses
// df^{ii}, as proteus does; the Jacobian of the interior flux uses the whole row df^{ij}.
//
// Two places where the engine is not proteus, on purpose: (1) proteus's adjoint
// (femIntegrals.c updateExteriorElementBoundaryDiffusionAdjoint) contracts the tensor as
// sum_{I,J} n_I a_{JI} grad w_I, the column sums of a, which equals (a^T n) . grad w only for a diagonal a;
// the engine uses (a^T n) . grad w. (2) proteus's Jacobian of the adjoint ignores da/du; so does the engine's,
// but its diffusive-flux Jacobian carries the da term like the element kernel. The regressions have an
// isotropic, constant a, where none of this shows.
//
// Output is per face (not scattered): Rf (nF, nc, nDOF) and Jf (nF, nc, nDOF, nc, nDOF); the driver adds them
// to the element rows of the face's element.
#pragma once
#include <cmath>
#include <cstring>
#include "masked.h"
#include "spaces.h"
#include "portable.h"

namespace adrh
{
  enum BCKind : int { Strong = 0, Flux = 1, WeakDirichlet = 2 };

  // What the BC functor fills in at one boundary quadrature point, per component.
  template <int nc>
  struct FaceBC
  {
    int kind[nc];
    double gD[nc];       // WeakDirichlet: the Dirichlet value
    double gdiff[nc];    // Flux: the prescribed diffusive flux -a grad u . n at u = 0 (0 = no diffusive flux),
    double dgdiff[nc];   //       plus dgdiff * u_i (Robin); 0 for a flux that does not depend on the trace
    bool hasAdv[nc];     // Flux: whether the advective flux is prescribed too,
    double fadv[nc];     //       and its value f . n
    ADRH_FN void reset()
    {
      for (int i = 0; i < nc; ++i)
      {
        kind[i] = Strong; gD[i] = 0.0; gdiff[i] = 0.0; dgdiff[i] = 0.0; hasAdv[i] = false; fadv[i] = 0.0;
      }
    }
  };

  // Penalty and adjoint parameters of the weak Dirichlet kinds (proteus's defaults per class).
  struct WeakDirichletParams
  {
    double penalty_constant, penalty_power, sigma;
    static WeakDirichletParams IIPG() { return {2.0, 1.0, 0.0}; }
    static WeakDirichletParams NIPG() { return {10.0, 1.0, -1.0}; }
    static WeakDirichletParams SIPG() { return {100.0, 1.0, 1.0}; }
  };

  // One exterior face, the function a Kokkos kernel calls per face. X (nDOF, 3) the DOF coordinates of the face's element, k the
  // element's local face index, F the face's number (what bc.evaluate gets, so that a functor can look a per-face kind up), ue
  // (nc, nDOF) the element's dofs; Rout (nc, nDOF) and Jout (nc, nDOF, nc, nDOF) out.
  template <class Coefficients, class BC, class Space = P1Simplex<Coefficients::nSpace>>
  ADRH_FN ADRH_FORCE_INLINE void exteriorFaceOne(const Coefficients &C, const BC &bc, const WeakDirichletParams &wd, int nQf, int F,
                                                int k, const double *X, const double *ue, const double *qpf, const double *qwf,
                                                double t, double *Rout, double *Jout)
  {
    using Co = Coefficients;
    constexpr int nc = Co::nc, nS = Co::nSpace, nD = Space::nDOF;
    static_assert(Space::nS == nS, "space dimension");
    PointGeometry<Space> G;
    Terms<nc, nS> T, Tb;
    // face diameter = the longest edge between its vertices (mesh.cpp computeGeometricInfo_*), for the penalty
    double h = 0.0;
    for (int p = 0; p < Space::nFaceVertices; ++p)
      for (int r = p + 1; r < Space::nFaceVertices; ++r)
      {
        const int vp = Space::faceVertex(k, p), vr = Space::faceVertex(k, r);
        double d2 = 0.0;
        for (int I = 0; I < 3; ++I)
        {
          const double d = X[vp * 3 + I] - X[vr * 3 + I];
          d2 += d * d;
        }
        h = std::fmax(h, std::sqrt(d2));
      }
    // pow(h, 1) is h exactly in a conforming libm, but not in every device math library: taking the power-one case out keeps the
    // penalty, and with it the Jacobian, the same bits on every backend
    const double penalty = wd.penalty_constant / (wd.penalty_power == 1.0 ? h : std::pow(h, wd.penalty_power));
    double Rloc[nc][nD] = {};
    double Jloc[nc][nD][nc][nD] = {};
    for (int q = 0; q < nQf; ++q)
    {
      double xi[nS];
      Space::faceToReference(k, qpf + q * (nS - 1), xi);
      G.compute(X, xi);
      const double *w = G.w, *x = G.x;
      const auto &gw = G.gw;
      double nrm[3], vn;
      G.faceNormal(k, nrm, vn);
      const double dS = qwf[q] * vn;
      double uq[nc], gu[nc][nS];
      for (int j = 0; j < nc; ++j)
      {
        uq[j] = 0.0;
        for (int I = 0; I < nS; ++I) gu[j][I] = 0.0;
        for (int b = 0; b < nD; ++b)
        {
          uq[j] += ue[j * nD + b] * w[b];
          for (int I = 0; I < nS; ++I) gu[j][I] += ue[j * nD + b] * gw[b][I];
        }
      }
      FaceBC<nc> B;
      B.reset();
      bc.evaluate(t, x, nrm, F, B);
      bool anyWeak = false;
      for (int i = 0; i < nc; ++i) anyWeak = anyWeak || (B.kind[i] == WeakDirichlet);
      zeroMasked<Co>(T);
      C.evaluate(t, x, uq, gu, T);
      // the boundary state: g_D where a component is WeakDirichlet, the interior value elsewhere
      double ub[nc];
      for (int j = 0; j < nc; ++j) ub[j] = (B.kind[j] == WeakDirichlet) ? B.gD[j] : uq[j];
      if (anyWeak)
      {
        zeroMasked<Co>(Tb);
        C.evaluate(t, x, ub, gu, Tb);
      }
      // grad phi^k = sum_j dphi[k][j] grad u_j
      double gphi[nc][nS];
      for (int kk = 0; kk < nc; ++kk)
        for (int I = 0; I < nS; ++I)
        {
          gphi[kk][I] = 0.0;
          for (int j = 0; j < nc; ++j) gphi[kk][I] += T.dphi[kk][j] * gu[j][I];
        }
      constexpr_for<0, nc>([&](auto i_) ADRH_LAMBDA_INLINE {
        constexpr int i = decltype(i_)::value;
        if (B.kind[i] == Strong) return;
        double Fn = 0.0;                 // the flux . n at this point
        double dFn[nc] = {};             // d Fn / d u_j, the part that multiplies w_b (no gradients)
        // advective part
        if constexpr (Co::has_advection && Co::f_nz(i))
        {
          double fl = 0.0, dl = 0.0;
          for (int I = 0; I < nS; ++I) fl += T.f[i][I] * nrm[I];
          if constexpr (Co::df_nz(i, i))
            for (int I = 0; I < nS; ++I) dl += T.df[i][i][I] * nrm[I];
          bool interior = false, prescribed = false;
          if (B.kind[i] == Flux)
          {
            if (B.hasAdv[i]) { prescribed = true; Fn += B.fadv[i]; }
            else interior = (dl >= 0.0);           // inflow without a value: dropped, as proteus does
          }
          else                                     // WeakDirichlet: upwind against f(g_D)
          {
            double fr = 0.0, dr = 0.0;
            for (int I = 0; I < nS; ++I) fr += Tb.f[i][I] * nrm[I];
            if constexpr (Co::df_nz(i, i))
              for (int I = 0; I < nS; ++I) dr += Tb.df[i][i][I] * nrm[I];
            if (dl >= 0.0 && dr >= 0.0) interior = true;
            else if (dl <= 0.0 && dr <= 0.0) interior = false;
            else if (dl >= 0.0 && dr <= 0.0)
            {
              const double uj = ub[i] - uq[i], fj = fr - fl;
              const double s = (std::fabs(uj) > std::fabs(fj) * 1.0e-16) ? fj / uj : 0.0;
              interior = (s >= 0.0);
            }
            else
            {                                      // transonic rarefaction: proteus averages the two
              Fn += 0.5 * (fl + fr);
              prescribed = true;                   // and takes half the interior speed as the Jacobian
              constexpr_for<0, nc>([&](auto j_) ADRH_LAMBDA_INLINE {
                constexpr int j = decltype(j_)::value;
                if constexpr (Co::df_nz(i, j))
                  for (int I = 0; I < nS; ++I) dFn[j] += 0.5 * T.df[i][j][I] * nrm[I];
              });
            }
            if (!prescribed && !interior) Fn += fr;
          }
          if (interior)
          {
            Fn += fl;
            constexpr_for<0, nc>([&](auto j_) ADRH_LAMBDA_INLINE {
              constexpr int j = decltype(j_)::value;
              if constexpr (Co::df_nz(i, j))
                for (int I = 0; I < nS; ++I) dFn[j] += T.df[i][j][I] * nrm[I];
            });
          }
        }
        // diffusive part
        if constexpr (Co::has_diffusion)
        {
          if (B.kind[i] == Flux)
          {
            Fn += B.gdiff[i] + B.dgdiff[i] * uq[i];
            dFn[i] += B.dgdiff[i];
          }
          else
          {
            constexpr_for<0, nc>([&](auto k_) ADRH_LAMBDA_INLINE {
              constexpr int kk = decltype(k_)::value;
              if constexpr (Co::a_nz(i, kk))
              {
                double an[nS];                     // (a^{ik})^T n
                for (int J = 0; J < nS; ++J)
                {
                  an[J] = 0.0;
                  for (int I = 0; I < nS; ++I) an[J] += nrm[I] * T.a[i][kk][I][J];
                }
                const double jump = uq[kk] - ub[kk];   // zero unless component kk is WeakDirichlet
                double qn = 0.0;
                for (int J = 0; J < nS; ++J) qn -= an[J] * gphi[kk][J];
                Fn += qn + penalty * jump;
                // Jacobian of -n . a grad phi^k + penalty (u_k - g) and of the adjoint, which carry grad w_b
                constexpr_for<0, nc>([&](auto j_) ADRH_LAMBDA_INLINE {
                  constexpr int j = decltype(j_)::value;
                  for (int b = 0; b < nD; ++b)
                  {
                    double d = 0.0;
                    if constexpr (Co::dphi_nz(kk, j))
                      for (int J = 0; J < nS; ++J) d -= an[J] * T.dphi[kk][j] * gw[b][J];
                    if constexpr (Co::da_nz(i, kk, j))
                      for (int I = 0; I < nS; ++I)
                        for (int J = 0; J < nS; ++J) d -= nrm[I] * T.da[i][kk][j][I][J] * gphi[kk][J] * w[b];
                    if (j == kk) d += penalty * w[b];
                    for (int a = 0; a < nD; ++a) Jloc[i][a][j][b] += dS * d * w[a];
                  }
                });
                if (wd.sigma != 0.0)
                {
                  for (int a = 0; a < nD; ++a)
                  {
                    double angw = 0.0;
                    for (int J = 0; J < nS; ++J) angw += an[J] * gw[a][J];
                    Rloc[i][a] -= dS * wd.sigma * jump * angw;
                    if (B.kind[kk] == WeakDirichlet)
                      for (int b = 0; b < nD; ++b) Jloc[i][a][kk][b] -= dS * wd.sigma * w[b] * angw;
                  }
                }
              }
            });
          }
        }
        for (int a = 0; a < nD; ++a)
        {
          Rloc[i][a] += dS * Fn * w[a];
          constexpr_for<0, nc>([&](auto j_) ADRH_LAMBDA_INLINE {
            constexpr int j = decltype(j_)::value;
            if (dFn[j] != 0.0)
              for (int b = 0; b < nD; ++b) Jloc[i][a][j][b] += dS * dFn[j] * w[b] * w[a];
          });
        }
      });
    }
    for (int kk = 0; kk < nc * nD; ++kk) Rout[kk] = (&Rloc[0][0])[kk];
    for (int kk = 0; kk < nc * nD * nc * nD; ++kk) Jout[kk] = (&Jloc[0][0][0][0])[kk];
  }

  // The boundary kernel, for any Lagrange Space (spaces.h; the default is P1 on simplices). bc.evaluate(t, x, n, F,
  // FaceBC<nc>&) is called per face F and quadrature point; the face index lets a functor look a per-face kind up.
  // Geometry (basis, gradients, outward normal, face measure) comes from the space's isoparametric map at each
  // face quadrature point, so the same code serves segments, triangles and, with a hexahedral space, quadrilaterals.
  template <class Coefficients, class BC, class Space = P1Simplex<Coefficients::nSpace>>
  void exteriorFaceKernel(const Coefficients &C, const BC &bc, const WeakDirichletParams &wd,
                          int nF, int nQf,
                          const int *faceLocal,            // (nF) local face index k (simplices: the opposite vertex)
                          const double *xn,                // (nF, nDOF, 3) DOF node coordinates of the face's element
                          const double *u,                 // (nF, nc, nDOF) the element's dofs
                          const double *qpf, const double *qwf,   // (nQf, nS-1), (nQf): quadrature on the reference face
                          double t, double *Rf, double *Jf)
  {
    constexpr int nc = Coefficients::nc, nD = Space::nDOF;
    for (int F = 0; F < nF; ++F)
      exteriorFaceOne<Coefficients, BC, Space>(C, bc, wd, nQf, F, faceLocal[F], xn + F * nD * 3, u + F * nc * nD, qpf, qwf, t,
                                               Rf + F * nc * nD, Jf + F * nc * nD * nc * nD);
  }

  // Boundary data of the sine solution u = sin(omega.x + omega0) for the regression, in nS dimensions. The kind
  // of each face comes from an array: 1 = Flux (diffusive flux given; advection leaves by the interior value,
  // inflow dropped), 2 = Flux with the advective flux (b.n) u given too, 3 = WeakDirichlet with u given,
  // 4 = Robin alpha u + beta q.n = g with g from the exact solution; anything else = Strong.
  template <int nS>
  struct SineBC
  {
    double b[3], kappa, omega[3], omega0, alpha, beta;
    const int *faceKind;
    ADRH_FN SineBC(const double *b_, double kappa_, const double *omega_, double omega0_, const int *faceKind_,
           double alpha_ = 1.0, double beta_ = 1.0)
        : kappa(kappa_), omega0(omega0_), alpha(alpha_), beta(beta_), faceKind(faceKind_)
    {
      for (int I = 0; I < 3; ++I) { b[I] = I < nS ? b_[I] : 0.0; omega[I] = I < nS ? omega_[I] : 0.0; }
    }
    ADRH_FN void evaluate(double, const double x[3], const double n[3], int F, FaceBC<1> &B) const
    {
      double arg = omega0, wn = 0.0, bn = 0.0;
      for (int I = 0; I < nS; ++I) { arg += omega[I] * x[I]; wn += omega[I] * n[I]; bn += b[I] * n[I]; }
      const double ue = std::sin(arg), qn = -kappa * wn * std::cos(arg);   // exact u and -kappa grad u . n
      switch (faceKind[F])
      {
      case 1: B.kind[0] = Flux; B.gdiff[0] = qn; break;
      case 2: B.kind[0] = Flux; B.gdiff[0] = qn; B.hasAdv[0] = true; B.fadv[0] = bn * ue; break;
      case 3: B.kind[0] = WeakDirichlet; B.gD[0] = ue; break;
      case 4:                                                                 // g = alpha u_exact + beta q_exact
        B.kind[0] = Flux; B.gdiff[0] = (alpha * ue + beta * qn) / beta; B.dgdiff[0] = -alpha / beta; break;
      default: B.kind[0] = Strong;
      }
    }
  };

  // The Gaussian of proteus's LinearADR_Decay_DiracIC (c = 0: LinearAD_DiracIC), the exact solution of
  // u_t + div(b u - a grad u) + c u = 0 with Dirac initial data on the whole space:
  //     u = u0 exp(-|y|^2 / (4 a t)) / (4 pi a t)^(nS/2) exp(-c t),   y = x - x0 - b t,   t = T + tStart,
  // and u = 0 where the exponent exceeds 100, as proteus's uOfXT has it; grad u = -u y / (2 a t).
  template <int nS>
  struct DiracExact
  {
    double b[3], a, c, tStart, u0, x0[3];
    ADRH_FN DiracExact(const double *b_, double a_, double c_, double tStart_, double u0_, const double *x0_)
        : a(a_), c(c_), tStart(tStart_), u0(u0_)
    {
      for (int I = 0; I < 3; ++I) { b[I] = I < nS ? b_[I] : 0.0; x0[I] = I < nS ? x0_[I] : 0.0; }
    }
    ADRH_FN double value(const double x[3], double T) const
    {
      const double t = T + tStart;
      double yy = 0.0;
      for (int I = 0; I < nS; ++I) { const double y = x[I] - x0[I] - b[I] * t; yy += y * y; }
      const double arg = yy / (4.0 * a * t);
      if (arg > 100.0) return 0.0;
      return u0 * std::exp(-arg) / std::pow(4.0 * a * M_PI * t, 0.5 * nS) * std::exp(-c * t);
    }
    ADRH_FN void gradient(const double x[3], double T, double g[3]) const
    {
      const double t = T + tStart, u = value(x, T);
      for (int I = 0; I < 3; ++I) g[I] = (I < nS) ? -u * (x[I] - x0[I] - b[I] * t) / (2.0 * a * t) : 0.0;
    }
  };

  // The BC functor fed by an exact solution (value and gradient at (x, t)) of a problem with advection b and an
  // isotropic diffusion kappa. The kind of each face comes from an array, as for SineBC: 1 = Flux (diffusive flux
  // given), 2 = Flux with the advective flux (b.n) u given too, 3 = WeakDirichlet, 4 = Robin alpha u + beta q.n = g.
  template <int nS, class Exact>
  struct ExactBC
  {
    Exact exact;
    double b[3], kappa, alpha, beta;
    const int *faceKind;
    ADRH_FN ExactBC(const Exact &exact_, const double *b_, double kappa_, const int *faceKind_, double alpha_ = 1.0, double beta_ = 1.0)
        : exact(exact_), kappa(kappa_), alpha(alpha_), beta(beta_), faceKind(faceKind_)
    {
      for (int I = 0; I < 3; ++I) b[I] = I < nS ? b_[I] : 0.0;
    }
    ADRH_FN void evaluate(double t, const double x[3], const double n[3], int F, FaceBC<1> &B) const
    {
      const double ue = exact.value(x, t);
      double g[3];
      exact.gradient(x, t, g);
      double gradn = 0.0, bn = 0.0;
      for (int I = 0; I < nS; ++I) { gradn += g[I] * n[I]; bn += b[I] * n[I]; }
      const double qn = -kappa * gradn;
      switch (faceKind[F])
      {
      case 1: B.kind[0] = Flux; B.gdiff[0] = qn; break;
      case 2: B.kind[0] = Flux; B.gdiff[0] = qn; B.hasAdv[0] = true; B.fadv[0] = bn * ue; break;
      case 3: B.kind[0] = WeakDirichlet; B.gD[0] = ue; break;
      case 4: B.kind[0] = Flux; B.gdiff[0] = (alpha * ue + beta * qn) / beta; B.dgdiff[0] = -alpha / beta; break;
      default: B.kind[0] = Strong;
      }
    }
  };
} // namespace adrh

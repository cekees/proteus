// Hand-written coefficient functors in the form a generator would emit (see adrh.h).
#pragma once
#include "masked.h"

namespace adrh
{
  // test/ci/poisson_3d_p.py: -div(a(x) grad u) = f(x), a = diag(x+5, y+5, z+10),
  // f = -2x - 2(5+x) - 2y - 2(5+y) - 2z - 2(10+z). Proteus stores r = -f.
  struct Poisson3D : DenseMasks
  {
    static constexpr int nc = 1, nSpace = 3;
    // only diffusion (a, phi = u) and the source r are present
    static constexpr bool has_mass = false, has_advection = false, has_hamiltonian = false;
    static constexpr bool dr_nz(int, int) { return false; }  // r does not depend on u
    void evaluate(double, const double x[3], const double u[1], const double gu[1][3], Terms<1, 3> &T) const
    {
      T.phi[0] = u[0];
      T.dphi[0][0] = 1.0;
      T.a[0][0][0][0] = x[0] + 5.0;
      T.a[0][0][1][1] = x[1] + 5.0;
      T.a[0][0][2][2] = x[2] + 10.0;
      const double f = -2.0 * x[0] - 2 * (5. + x[0]) - 2. * x[1] - 2. * (5. + x[1]) - 2. * x[2] - 2. * (10 + x[2]);
      T.r[0] = -f;
    }
  };

  // A coupled two-component system with every term nonlinear, for checking the Jacobian by
  // finite differences. The potentials are linear (phi = u), as in the existing kernels' Jacobians.
  struct Coupled2 : DenseMasks
  {
    static constexpr int nc = 2, nSpace = 3;
    // the coupling is spelled out: dm[0][1], df[0][1], a^{01}, H_0 depends on grad u_0 and grad u_1 only
    static constexpr bool dm_nz(int i, int j) { return i == j || (i == 0 && j == 1); }
    static constexpr bool df_nz(int i, int j) { return i == j || (i == 0 && j == 1); }
    static constexpr bool a_nz(int i, int k) { return i == k || (i == 0 && k == 1); }
    static constexpr bool da_nz(int i, int k, int j) { return (i == 0 && k == 0 && j == 0) || (i == 1 && k == 1 && j == 1) || (i == 0 && k == 1 && j == 0); }
    static constexpr bool dphi_nz(int k, int j) { return k == j; }
    static constexpr bool H_nz(int i) { return i == 0; }
    static constexpr bool dH_nz(int i, int j) { return i == 0; }
    void evaluate(double t, const double x[3], const double u[2], const double gu[2][3], Terms<2, 3> &T) const
    {
      const double u0 = u[0], u1 = u[1];
      // mass
      T.m[0] = u0 + 0.2 * u1 * u1;  T.dm[0][0] = 1.0;  T.dm[0][1] = 0.4 * u1;
      T.m[1] = u1;                  T.dm[1][1] = 1.0;
      // advection
      const double c0[3] = {1.0, 0.5, 0.0}, c1[3] = {0.0, 0.0, 1.0};
      for (int I = 0; I < 3; ++I)
      {
        T.f[0][I] = c0[I] * u0 * u1;   T.df[0][0][I] = c0[I] * u1;  T.df[0][1][I] = c0[I] * u0;
        T.f[1][I] = c1[I] * u1 * u1;   T.df[1][1][I] = 2.0 * c1[I] * u1;
      }
      // diffusion: phi_k = u_k; a^{00}, a^{11} isotropic in u, a^{01} couples grad u1 into equation 0
      T.phi[0] = u0; T.dphi[0][0] = 1.0;
      T.phi[1] = u1; T.dphi[1][1] = 1.0;
      for (int I = 0; I < 3; ++I)
      {
        T.a[0][0][I][I] = 1.0 + 0.3 * u0 * u0;   T.da[0][0][0][I][I] = 0.6 * u0;
        T.a[1][1][I][I] = 0.5 + 0.1 * u1 * u1 + 0.05 * x[0];  T.da[1][1][1][I][I] = 0.2 * u1;
        T.a[0][1][I][I] = 0.05 * u0 + 0.01 * x[1];  T.da[0][1][0][I][I] = 0.05;
      }
      // Hamiltonian: depends on the gradients only
      T.H[0] = 0.4 * (1.0 * gu[0][0] + 2.0 * gu[0][1]) + 0.1 * (gu[1][0] * gu[1][0] + gu[1][1] * gu[1][1] + gu[1][2] * gu[1][2]);
      T.dH[0][0][0] = 0.4; T.dH[0][0][1] = 0.8;
      for (int I = 0; I < 3; ++I) T.dH[0][1][I] = 0.2 * gu[1][I];
      // reaction
      T.r[0] = u0 * u1 - 0.3 * std::sin(t + x[2]);  T.dr[0][0] = u1;  T.dr[0][1] = u0;
      T.r[1] = u1 * u1 * u1 - u0;                   T.dr[1][1] = 3.0 * u1 * u1;  T.dr[1][0] = -1.0;
    }
  };

  // Five components in a chain, for the cost of ignoring structure: mass, advection and diffusion are
  // diagonal in the components, the reaction couples neighbours (tridiagonal, 13 of 25 blocks).
  // UseMasks = false inherits the all-true masks and writes exactly the same values, so the two
  // differ only in the work the kernel does.
  template <bool UseMasks>
  struct Chain5 : DenseMasks
  {
    static constexpr int nc = 5, nSpace = 3;
    static constexpr bool has_hamiltonian = !UseMasks;  // there is no Hamiltonian; dense still loops over it
    static constexpr bool m_nz(int i) { return !UseMasks || true; }
    static constexpr bool dm_nz(int i, int j) { return !UseMasks || i == j; }
    static constexpr bool df_nz(int i, int j) { return !UseMasks || i == j; }
    static constexpr bool a_nz(int i, int k) { return !UseMasks || i == k; }
    static constexpr bool da_nz(int i, int k, int j) { return !UseMasks || (i == k && k == j); }
    static constexpr bool dphi_nz(int k, int j) { return !UseMasks || k == j; }
    static constexpr bool H_nz(int i) { return !UseMasks; }
    static constexpr bool dH_nz(int i, int j) { return !UseMasks; }
    static constexpr bool dr_nz(int i, int j) { return !UseMasks || (i - j >= -1 && i - j <= 1); }
    void evaluate(double, const double x[3], const double u[5], const double gu[5][3], Terms<5, 3> &T) const
    {
      const double v[3] = {1.0, 0.5, 0.25};
      for (int i = 0; i < 5; ++i)
      {
        T.m[i] = u[i];                 T.dm[i][i] = 1.0;
        for (int I = 0; I < 3; ++I) { T.f[i][I] = v[I] * u[i]; T.df[i][i][I] = v[I]; }
        T.phi[i] = u[i];               T.dphi[i][i] = 1.0;
        const double kappa = 0.1 * (i + 1) * (1.0 + 0.1 * u[i] * u[i]);
        for (int I = 0; I < 3; ++I) { T.a[i][i][I][I] = kappa; T.da[i][i][i][I][I] = 0.02 * (i + 1) * u[i]; }
        const double um = i > 0 ? u[i - 1] : 0.0, up = i < 4 ? u[i + 1] : 0.0;
        T.r[i] = um * u[i] - u[i] * up + 0.1 * u[i] + 0.01 * x[0];
        T.dr[i][i] = um - up + 0.1;
        if (i > 0) T.dr[i][i - 1] = u[i];
        if (i < 4) T.dr[i][i + 1] = -u[i];
      }
    }
  };

  // Steady linear advection-diffusion with the manufactured sine solution of
  // proteus.AnalyticalSolutions.LinearADR_Sine (c = 0):
  //   div(b u - kappa grad u) + d(x) = 0,  u = sin(omega.x + omega0),
  //   d(x) = -(b.omega) cos(omega.x + omega0) - kappa |omega|^2 sin(omega.x + omega0).
  // In proteus's form f = b u, a = kappa I, phi = u, r = d(x). The parameters are data of the functor.
  // Steady advection-diffusion with the manufactured solution u = sin(omega.x + omega0) in nS = 2 or 3 dimensions:
  // div(b u - kappa grad u) + r(x) = 0 with r chosen so u is exact (AnalyticalSolutions.LinearADR_Sine, c = 0).
  template <int nS_>
  struct AdvectionDiffusionSine : DenseMasks
  {
    static constexpr int nc = 1, nSpace = nS_;
    static constexpr bool has_mass = false, has_hamiltonian = false;
    static constexpr bool da_nz(int, int, int) { return false; }  // a does not depend on u
    static constexpr bool dr_nz(int, int) { return false; }       // nor does the source
    double b[3], kappa, omega[3], omega0;
    AdvectionDiffusionSine(const double *b_, double kappa_, const double *omega_, double omega0_)
        : kappa(kappa_), omega0(omega0_)
    {
      for (int I = 0; I < 3; ++I) { b[I] = I < nS_ ? b_[I] : 0.0; omega[I] = I < nS_ ? omega_[I] : 0.0; }
    }
    void evaluate(double, const double x[3], const double u[1], const double gu[1][nS_], Terms<1, nS_> &T) const
    {
      T.phi[0] = u[0];
      T.dphi[0][0] = 1.0;
      double bw = 0.0, ww = 0.0, arg = omega0;
      for (int I = 0; I < nS_; ++I) { bw += b[I] * omega[I]; ww += omega[I] * omega[I]; arg += omega[I] * x[I]; }
      for (int I = 0; I < nS_; ++I)
      {
        T.f[0][I] = b[I] * u[0];
        T.df[0][0][I] = b[I];
        T.a[0][0][I][I] = kappa;
      }
      T.r[0] = -bw * std::cos(arg) - kappa * ww * std::sin(arg);
    }
  };
} // namespace adrh

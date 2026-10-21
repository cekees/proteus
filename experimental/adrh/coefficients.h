// Hand-written coefficient functors in the form a generator would emit (see adrh.h).
#pragma once
#include "adrh.h"

namespace adrh
{
  // test/ci/poisson_3d_p.py: -div(a(x) grad u) = f(x), a = diag(x+5, y+5, z+10),
  // f = -2x - 2(5+x) - 2y - 2(5+y) - 2z - 2(10+z). Proteus stores r = -f.
  struct Poisson3D
  {
    static constexpr int nc = 1, nSpace = 3;
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
  struct Coupled2
  {
    static constexpr int nc = 2, nSpace = 3;
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
} // namespace adrh

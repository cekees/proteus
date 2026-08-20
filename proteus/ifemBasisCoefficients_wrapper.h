#ifndef IFEM_BASIS_COEFFICIENTS_WRAPPER_H
#define IFEM_BASIS_COEFFICIENTS_WRAPPER_H

#include <array>
#include <cmath>
#include <stdexcept>

namespace proteus
{
  // Solve A*x = b for a small, fixed-size N x N system via Gauss-Jordan
  // elimination with partial pivoting. A is row-major and destroyed.
  template <int N>
  inline void solveSmallLinearSystem(double A[N * N], double b[N], double x[N])
  {
    for (int col = 0; col < N; col++)
    {
      int piv = col;
      double maxval = fabs(A[col * N + col]);
      for (int r = col + 1; r < N; r++)
      {
        if (fabs(A[r * N + col]) > maxval)
        {
          maxval = fabs(A[r * N + col]);
          piv = r;
        }
      }
      if (piv != col)
      {
        for (int j = 0; j < N; j++)
          std::swap(A[col * N + j], A[piv * N + j]);
        std::swap(b[col], b[piv]);
      }
      const double pivval = A[col * N + col];
      assert(fabs(pivval) > 1.0e-14 && "Singular system in solveSmallLinearSystem");
      const double invPiv = 1.0 / pivval;
      for (int j = col; j < N; j++)
        A[col * N + j] *= invPiv;
      b[col] *= invPiv;
      for (int r = 0; r < N; r++)
      {
        if (r == col)
          continue;
        const double factor = A[r * N + col];
        if (factor != 0.0)
        {
          for (int j = col; j < N; j++)
            A[r * N + j] -= factor * A[col * N + j];
          b[r] -= factor * b[col];
        }
      }
    }
    for (int i = 0; i < N; i++)
      x[i] = b[i];
  }

  // P1 linear basis solver (overload for 3 nodal values)
  //
  // Solves the same constraint system as
  // proteus.ifemBasisCoefficients._solveCoefficients_P1 (nodal
  // interpolation + interface continuity + flux jump), but numerically:
  // the constraint matrix depends only on (x0,y0,nx,ny,ma,mb) -- not on
  // nodal_values -- so this is a plain 6x6 linear solve with no symbolic
  // differentiation and no Python round-trip.
  inline std::array<double, 6> solve_ifem_basis_coefficients(
      int basis_order,
      double x0,
      double y0,
      double nx,
      double ny,
      double ma,
      double mb,
      double jf,
      double Jit00,
      double Jit01,
      double Jit10,
      double Jit11,
      const std::array<double, 3> &nodal_values)
  {
    (void)basis_order; // always 1 for this overload
    const double tx = Jit00 * nx + Jit10 * ny;
    const double ty = Jit01 * nx + Jit11 * ny;

    // unknowns: [a1,a2,a3,b1,b2,b3]
    double A[36] = {
        1., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 1., 0.,
        0., 0., 0., 1., 0., 1.,
        -1., -x0, 0., 1., x0, 0.,
        -1., 0., -y0, 1., 0., y0,
        0., -ma * tx, -ma * ty, 0., mb * tx, mb * ty};
    double rhs[6] = {nodal_values[0], nodal_values[1], nodal_values[2], 0.0, 0.0, jf};

    double coeffs[6];
    solveSmallLinearSystem<6>(A, rhs, coeffs);

    std::array<double, 6> out;
    for (int i = 0; i < 6; ++i)
    {
      out[i] = coeffs[i];
      if (std::isnan(out[i]) || std::isinf(out[i]))
        throw std::runtime_error("solve_ifem_basis_coefficients (P1) returned NaN or Inf");
    }
    return out;
  }

  // P2 quadratic basis solver (overload for 6 nodal values)
  //
  // Solves the same constraint system as
  // proteus.ifemBasisCoefficients._solveCoefficients_P2 numerically. va/vb
  // are full 2D quadratics, so all constraints (nodal interpolation,
  // interface continuity, flux jump, normal-Laplacian jump) reduce to
  // linear equations in the 12 coefficients with coefficients that are
  // plain algebraic functions of (x0,y0,nx,ny,ma,mb) -- the "symbolic
  // differentiation" in the Python version is just picking off constant
  // polynomial coefficients, so it never needed to happen at runtime.
  inline std::array<double, 12> solve_ifem_basis_coefficients(
      int basis_order,
      double x0,
      double y0,
      double nx,
      double ny,
      double ma,
      double mb,
      double jf,
      double Jit00,
      double Jit01,
      double Jit10,
      double Jit11,
      const std::array<double, 6> &nodal_values)
  {
    (void)basis_order; // always 2 for this overload
    assert(!((x0 == 0.0 && y0 == 0.0) || (x0 == 1.0 && y0 == 0.0) || (x0 == 0.0 && y0 == 1.0)) &&
           "Interface passes through a triangle vertex");
    assert(!(x0 > 0.5 && y0 <= 0.5) &&
           "Invalid interface location for quadratic basis functions (should have been flipped upstream)");

    const double tx = Jit00 * nx + Jit10 * ny;
    const double ty = Jit01 * nx + Jit11 * ny;
    const double v1 = nodal_values[0], v2 = nodal_values[1], v3 = nodal_values[2],
                 v4 = nodal_values[3], v5 = nodal_values[4], v6 = nodal_values[5];

    // unknowns: [a1,a2,a3,a4,a5,a6, b1,b2,b3,b4,b5,b6]
    double A[144] = {0.0};
    double rhs[12] = {0.0};

    // c1: va(0,0) = v1
    A[0 * 12 + 0] = 1.0;
    rhs[0] = v1;

    // c2: vb(1,0) = v2  (x^2 term is 1 at x=1, so b5 contributes)
    A[1 * 12 + 6] = 1.0;
    A[1 * 12 + 7] = 1.0;
    A[1 * 12 + 10] = 1.0;
    rhs[1] = v2;

    // c3: vb(0,1) = v3  (y^2 term is 1 at y=1, so b6 contributes)
    A[2 * 12 + 6] = 1.0;
    A[2 * 12 + 8] = 1.0;
    A[2 * 12 + 11] = 1.0;
    rhs[2] = v3;

    // c4: vb(1/2,0) = v4 if x0<=1/2, else va(1/2,0) = v4
    if (x0 <= 0.5)
    {
      A[3 * 12 + 6] = 1.0;
      A[3 * 12 + 7] = 0.5;
      A[3 * 12 + 10] = 0.25;
    }
    else
    {
      A[3 * 12 + 0] = 1.0;
      A[3 * 12 + 1] = 0.5;
      A[3 * 12 + 4] = 0.25;
    }
    rhs[3] = v4;

    // c5: vb(1/2,1/2) = v5
    A[4 * 12 + 6] = 1.0;
    A[4 * 12 + 7] = 0.5;
    A[4 * 12 + 8] = 0.5;
    A[4 * 12 + 9] = 0.25;
    A[4 * 12 + 10] = 0.25;
    A[4 * 12 + 11] = 0.25;
    rhs[4] = v5;

    // c6: vb(0,1/2) = v6 if y0<=1/2, else va(0,1/2) = v6
    if (y0 <= 0.5)
    {
      A[5 * 12 + 6] = 1.0;
      A[5 * 12 + 8] = 0.5;
      A[5 * 12 + 11] = 0.25;
    }
    else
    {
      A[5 * 12 + 0] = 1.0;
      A[5 * 12 + 2] = 0.5;
      A[5 * 12 + 5] = 0.25;
    }
    rhs[5] = v6;

    // c7: vb(x0,0) - va(x0,0) = 0
    A[6 * 12 + 0] = -1.0;
    A[6 * 12 + 1] = -x0;
    A[6 * 12 + 4] = -x0 * x0;
    A[6 * 12 + 6] = 1.0;
    A[6 * 12 + 7] = x0;
    A[6 * 12 + 10] = x0 * x0;

    // c8: vb(0,y0) - va(0,y0) = 0
    A[7 * 12 + 0] = -1.0;
    A[7 * 12 + 2] = -y0;
    A[7 * 12 + 5] = -y0 * y0;
    A[7 * 12 + 6] = 1.0;
    A[7 * 12 + 8] = y0;
    A[7 * 12 + 11] = y0 * y0;

    // c9: vb(x0/2,y0/2) - va(x0/2,y0/2) = 0
    A[8 * 12 + 0] = -1.0;
    A[8 * 12 + 1] = -0.5 * x0;
    A[8 * 12 + 2] = -0.5 * y0;
    A[8 * 12 + 3] = -0.25 * x0 * y0;
    A[8 * 12 + 4] = -0.25 * x0 * x0;
    A[8 * 12 + 5] = -0.25 * y0 * y0;
    A[8 * 12 + 6] = 1.0;
    A[8 * 12 + 7] = 0.5 * x0;
    A[8 * 12 + 8] = 0.5 * y0;
    A[8 * 12 + 9] = 0.25 * x0 * y0;
    A[8 * 12 + 10] = 0.25 * x0 * x0;
    A[8 * 12 + 11] = 0.25 * y0 * y0;

    // c10: mb*flux_b(x0,0) - ma*flux_a(x0,0) = jf
    A[9 * 12 + 1] = -ma * tx;
    A[9 * 12 + 2] = -ma * ty;
    A[9 * 12 + 3] = -ma * x0 * ty;
    A[9 * 12 + 4] = -ma * 2.0 * x0 * tx;
    A[9 * 12 + 7] = mb * tx;
    A[9 * 12 + 8] = mb * ty;
    A[9 * 12 + 9] = mb * x0 * ty;
    A[9 * 12 + 10] = mb * 2.0 * x0 * tx;
    rhs[9] = jf;

    // c11: mb*flux_b(0,y0) - ma*flux_a(0,y0) = jf
    A[10 * 12 + 1] = -ma * tx;
    A[10 * 12 + 2] = -ma * ty;
    A[10 * 12 + 3] = -ma * y0 * tx;
    A[10 * 12 + 5] = -ma * 2.0 * y0 * ty;
    A[10 * 12 + 7] = mb * tx;
    A[10 * 12 + 8] = mb * ty;
    A[10 * 12 + 9] = mb * y0 * tx;
    A[10 * 12 + 11] = mb * 2.0 * y0 * ty;
    rhs[10] = jf;

    // c12: mb*vb_nn(x0/2,y0/2) - ma*va_nn(x0/2,y0/2) = 0
    // (va_nn/vb_nn are constant over the element for a quadratic, so the
    // evaluation point doesn't actually matter)
    A[11 * 12 + 3] = -ma * 2.0 * tx * ty;
    A[11 * 12 + 4] = -ma * 2.0 * tx * tx;
    A[11 * 12 + 5] = -ma * 2.0 * ty * ty;
    A[11 * 12 + 9] = mb * 2.0 * tx * ty;
    A[11 * 12 + 10] = mb * 2.0 * tx * tx;
    A[11 * 12 + 11] = mb * 2.0 * ty * ty;

    double coeffs[12];
    solveSmallLinearSystem<12>(A, rhs, coeffs);

    std::array<double, 12> out;
    for (int i = 0; i < 12; ++i)
    {
      out[i] = coeffs[i];
      if (std::isnan(out[i]) || std::isinf(out[i]))
        throw std::runtime_error("solve_ifem_basis_coefficients (P2) returned NaN or Inf");
    }
    return out;
  }
}

#endif

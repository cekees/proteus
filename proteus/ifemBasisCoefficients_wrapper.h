#ifndef IFEM_BASIS_COEFFICIENTS_WRAPPER_H
#define IFEM_BASIS_COEFFICIENTS_WRAPPER_H

#include <array>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"

namespace proteus
{
  namespace py = pybind11;

  // P1 linear basis solver (overload for 3 nodal values)
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
    try {
      py::gil_scoped_acquire gil;

      static py::object solve_func;
      if (!solve_func)
      {
        py::module_ mod = py::module_::import("proteus.ifemBasisCoefficients");
        solve_func = mod.attr("solveCoefficients");
      }

      // Convert nx and ny to single precision (float) only
      float nx_f = static_cast<float>(nx);
      float ny_f = static_cast<float>(ny);

      py::list nodes;
      for (double v : nodal_values)
        nodes.append(v);

      py::object result = solve_func(basis_order, x0, y0, nx_f, ny_f, ma, mb, jf, Jit00, Jit01, Jit10, Jit11, nodes);
      
      if (result.is_none()) {
        throw std::runtime_error("solveCoefficients returned None");
      }
      
      py::sequence seq = result.cast<py::sequence>();

      if (seq.size() != 6)
        throw std::runtime_error("solveCoefficients must return 6 coefficients");

      std::array<double, 6> out;
      for (size_t i = 0; i < 6; ++i) {
        out[i] = py::float_(seq[i]);
        if (std::isnan(out[i]) || std::isinf(out[i])) {
          throw std::runtime_error("solveCoefficients returned NaN or Inf");
        }
      }

      return out;
    }
    catch (const py::error_already_set &e) {
      std::cout << "Python error in solveCoefficients: " << e.what() << std::endl << std::flush;
      throw;
    }
    catch (const std::exception &e) {
      std::cout << "C++ error in solve_ifem_basis_coefficients: " << e.what() << std::endl << std::flush;
      throw;
    }
  }

  // P2 quadratic basis solver (overload for 6 nodal values)
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
    try {
      py::gil_scoped_acquire gil;

      static py::object solve_func;
      if (!solve_func)
      {
        py::module_ mod = py::module_::import("proteus.ifemBasisCoefficients");
        solve_func = mod.attr("solveCoefficients");
      }

      // Convert nx and ny to single precision (float) only
      float nx_f = static_cast<float>(nx);
      float ny_f = static_cast<float>(ny);

      py::list nodes;
      for (double v : nodal_values)
        nodes.append(v);

      py::object result = solve_func(basis_order, x0, y0, nx_f, ny_f, ma, mb, jf, Jit00, Jit01, Jit10, Jit11, nodes);
      
      if (result.is_none()) {
        throw std::runtime_error("solveCoefficients returned None");
      }
      
      py::sequence seq = result.cast<py::sequence>();

      if (seq.size() != 12)
        throw std::runtime_error("solveCoefficients must return 12 coefficients");

      std::array<double, 12> out;
      for (size_t i = 0; i < 12; ++i) {
        out[i] = py::float_(seq[i]);
        if (std::isnan(out[i]) || std::isinf(out[i])) {
          throw std::runtime_error("solveCoefficients returned NaN or Inf");
        }
      }

      return out;
    }
    catch (const py::error_already_set &e) {
      std::cout << "Python error in solveCoefficients: " << e.what() << std::endl << std::flush;
      throw;
    }
    catch (const std::exception &e) {
      std::cout << "C++ error in solve_ifem_basis_coefficients: " << e.what() << std::endl << std::flush;
      throw;
    }
  }
}

#endif

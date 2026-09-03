#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#define FORCE_IMPORT_ARRAY
#include "m_comp_co2.h"

#if defined(__GNUC__) && !defined(__clang__)
namespace workaround
{
inline void define_allocators()
{
  std::allocator<int>    a0;
  std::allocator<double> a1;
}
} // namespace workaround
#endif

namespace py = pybind11;
using proteus::m_comp_co2::M_comp_co2_base;

PYBIND11_MODULE(cm_comp_co2, m)
{
  xt::import_numpy();

  py::class_<M_comp_co2_base>(m, "cM_comp_co2_base")
    .def(py::init(&proteus::m_comp_co2::newm_comp_co2))
    .def("calculateResidual", &M_comp_co2_base::calculateResidual)
    .def("calculateJacobian", &M_comp_co2_base::calculateJacobian)
    .def("invert", &M_comp_co2_base::invert)
    .def("FCTStep", &M_comp_co2_base::FCTStep)
    //.def("kth_FCT_step", &M_comp_co2_base::kth_FCT_step)
    .def("calculateResidual_entropy_viscosity", &M_comp_co2_base::calculateResidual_entropy_viscosity)
    .def("calculateMassMatrix", &M_comp_co2_base::calculateMassMatrix)
    .def("dissolutionFlash", &M_comp_co2_base::dissolutionFlash)
    .def("calculateFlashFields", &M_comp_co2_base::calculateFlashFields);
}
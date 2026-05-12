#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#define FORCE_IMPORT_ARRAY
#include "mphase_co2.h"

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
using proteus::mphase_co2::Mphase_co2_base;

PYBIND11_MODULE(cmphase_co2, m)
{
  xt::import_numpy();

  py::class_<Mphase_co2_base>(m, "cMphase_co2_base")
    .def(py::init(&proteus::mphase_co2::newmphase_co2))
    .def("calculateResidual", &Mphase_co2_base::calculateResidual)
    .def("calculateJacobian", &Mphase_co2_base::calculateJacobian)
    .def("invert", &Mphase_co2_base::invert)
    .def("FCTStep", &Mphase_co2_base::FCTStep)
    .def("FCTStep_n", &Mphase_co2_base::FCTStep_n)
    //.def("kth_FCT_step", &Mphase_co2_base::kth_FCT_step)
    .def("calculateResidual_entropy_viscosity", &Mphase_co2_base::calculateResidual_entropy_viscosity)
    .def("calculateMassMatrix", &Mphase_co2_base::calculateMassMatrix);
}
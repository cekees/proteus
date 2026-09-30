#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "ADR.h"

namespace py = pybind11;
using proteus::richards::ADR_base;

PYBIND11_MODULE(cADR, m)
{
    proteus::import_numpy();

    py::class_<ADR_base>(m, "cADR_base")
      .def(py::init(&proteus::richards::newADR))
      .def("calculateResidual", &ADR_base::calculateResidual)
      .def("calculateJacobian", &ADR_base::calculateJacobian);
}

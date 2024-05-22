#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#define FORCE_IMPORT_ARRAY
#include "ADR.h"

#if defined(__GNUC__) && !defined(__clang__)
    namespace workaround
    {
        inline void define_allocators()
        {
            std::allocator<int> a0;
            std::allocator<double> a1;
        }
    }
#endif

namespace py = pybind11;
using proteus::ADR_base;

PYBIND11_MODULE(cADR, m)
{
    xt::import_numpy();

    py::class_<ADR_base>(m, "cADR_base")
        .def(py::init(&proteus::newADR))
        .def("calculateResidual", &ADR_base::calculateResidual)
        .def("calculateJacobian", &ADR_base::calculateJacobian)
        .def("invert", &ADR_base::invert)
        .def("FCTStep", &ADR_base::FCTStep)
        .def("kth_FCT_step", &ADR_base::kth_FCT_step)
        .def("calculateResidual_entropy_viscosity", &ADR_base::calculateResidual_entropy_viscosity)
        .def("calculateMassMatrix", &ADR_base::calculateMassMatrix);
}

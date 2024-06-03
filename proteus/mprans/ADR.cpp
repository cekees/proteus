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
using proteus::mprans::ADR_base;

PYBIND11_MODULE(cVOF, m)
{
    xt::import_numpy();

    py::class_<ADR_base>(m, "cADR_base")
      .def(py::init(&proteus::mprans::newADR))
        .def("calculateResidualElementBased"    , &ADR_base::calculateResidualElementBased  )
        .def("calculateJacobian"                , &ADR_base::calculateJacobian              )
        .def("FCTStep"                          , &ADR_base::FCTStep                        )
        .def("calculateResidualEdgeBased"       , &ADR_base::calculateResidualEdgeBased     );
}

"""
Modules for computing numerical solutions of differential equations
"""
import os as _os
import sys as _sys

# On macOS, libfabric (which conda-forge's MPICH runs over) picks its
# "sockets" provider, whose two connection threads busy-poll at 100% CPU
# each from MPI_Init on, in every process: each MPI rank holds three cores,
# half of it system time. Its "tcp" provider has no such threads. This has
# to be set before MPI starts, which importing proteus.Comm (mpi4py.MPI)
# does, and it only takes effect if MPI wasn't started before proteus was
# imported. A value already in the environment is left alone; libraries
# other than libfabric ignore the variable.
if _sys.platform == "darwin":
    _os.environ.setdefault("FI_PROVIDER", "tcp")

__version__ = '2.0.0.dev'

__all__ = ["Archiver",
           "Domain",
           "InputTranslators",
           "SplitOperator",
           "StepControl",
           "NumericalSolution",
           "Comm",
           "AnalyticalSolutions",
           "TransportCoefficients",
           "TwophaseDarcyCoefficients",
           "DiagUtils",
           "EGeometry",
           "ErrorEstimators",
           "FemTools",
           "LatexReport",
           "LinearAlgebraTools",
           "LinearSolvers",
           "MeshTools",
           "NonlinearSolvers",
           "Norms",
           "NumericalFlux",
           "PostProcessingTools",
           "Profiling",
           "Quadrature",
           "RefUtils",
           "ShockCapturing",
           "SimTools",
           "SubgridError",
           "SubsurfaceTransportCoefficients",
           "TimeIntegration",
           "Transport",
           "Viewers",
           "AuxiliaryVariables",
           "deim_utils",
           "default_p",
           "default_n",
           "default_s",
           "default_so",
           "InputTranslators",
           "cfemIntegrals",
           "cshockCapturing",
           "csmoothers",
           "csubgridError",
           "ctimeIntegration",
           "ctransportCoefficients",
           "clapack",
           "superluWrappers",
           "cmeshTools",
           "cnumericalFlux",
           "cTwophaseDarcyCoefficients",
           "cADR",
           "deim_utils",
           "WaveTools",
           "Context",
           "BoundaryConditions",
           "SpatialTools",
           "defaults"]

from proteus import *
import numpy
from math import *

details = """
The Goswami-Clement Saltwater Intrusion Verification Problem (Goswami-Clement,2007)
Variable Density Groundwater Flow and Transport
in homogeneous porous medium
The primitive variables are the mass fraction (mf) 
and the freshwater pressure head (h)
"""
#spatial domain Omega = [0,53 cm] x [0,26 cm]
# SHOCK CAPTURING PARAMETERS #
shockCapturingFactor_tadr=0.2
lag_shockCapturing_tadr=True

# number of space dimensions #
#nd=ct.nd

# General parameters #
parallel = True # if True use PETSc solvers
linearSmoother = None
checkMass = False

# Finite element sapce #
pDegree_tadr=1
useBernstein=False
useHex=False

# quadrature order #
tadr_quad_order = 2*pDegree_tadr+1

# parallel partitioning info #
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node

L=(.530,.26,1)#.26 
nd = 2

nnx =107#213#107
nny =53#105#53#53# 105

#if (unstructured):
#for unstructured domains
domain = Domain.RectangularDomain(L=[L[0],L[1]],name="gwvd_gc_domain",units="m")
polyfile="gwvd_gc_domain_2d"
domain.writePoly(polyfile)
triangleOptions = "q30Dena0.00001"

#domain.MeshOptions.nn = nn
domain.MeshOptions.nnx = nnx
domain.MeshOptions.nny = nny
#domain.MeshOptions.nnz = nnz
domain.MeshOptions.triangleFlag=0

soname="gw_cl"

class MyCoefficients(TADR.Coefficients):
    def attachModels(self, modelList):
        """
        Attach the Richards model to TADR and pass the velocity field (grad(u)).
        """
        # Attach Richards model
        self.model = modelList[0]  # Richards model
        self.rdModel = self.model  # Alias for convenience
            # Log modelList for inspection
        logEvent(f"Model list contains {len(modelList)} models:")
        for i, model in enumerate(modelList):
            logEvent(f"Model {i}: {model.name}, Type: {type(model)}")

        # Reference velocity fields directly from Richards model
        if ('grad(u_v)', 0) in self.model.q:
            self.q_v = self.model.q[('grad(u_v)', 0)]  # Reference gradient of u (velocity)
        else:
            raise KeyError("Richards model q[('grad(u_v)', 0)] not found!")

        if ('grad(u)', 0) in self.model.ebqe:
            self.ebqe_v = self.model.ebqe[('grad(u)', 0)]  # Reference boundary gradient of u (velocity)
        else:
            raise KeyError("Richards model ebqe[('grad(u)', 0)] not found!")

        # Ensure quadrature volumes exist
        if 'dV' not in self.model.q:
            self.model.q['dV'] = np.ones(self.model.q[('u', 0)].shape, 'd')

        if 'dV_last' not in self.model.q:
            self.model.q['dV_last'] = np.zeros_like(self.model.q['dV'])

        # Log for debugging
        logEvent(f"Richards velocity (q_v): mean={self.q_v.mean()}, min={self.q_v.min()}, max={self.q_v.max()}")
        logEvent(f"Richards boundary velocity (ebqe_v): mean={self.ebqe_v.mean()}, min={self.ebqe_v.min()}, max={self.ebqe_v.max()}")



from proteus import *
from proteus.default_p import *

nd = 2

# BASE case geometry from the reference setup:
# 40 m buffer + 20 m inlet strip + 40 m buffer, with 10 m total depth.
L = (100.0, 50.0)

domain = Domain.RectangularDomain(L=[L[0], L[1]],
                                  name="watertable_4m_domain",
                                  units="m")
polyfile = "watertable_4m_domain_2d"
domain.writePoly(polyfile)

boundaryTags = domain.boundaryTags

# Keep the mesh anisotropy moderate while giving the vertical direction
# enough resolution to capture unstable fingers below the infiltration strip.
domain.MeshOptions.nnx = 401
domain.MeshOptions.nny = 201
domain.MeshOptions.triangleFlag = 0
#triangleOptions = "q30Dena0.002778"
triangleOptions = "q30Dena0.001246"
0.001246
femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexLobattoQuadrature(nd, 1)
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd - 1, 1)
subgridError = None

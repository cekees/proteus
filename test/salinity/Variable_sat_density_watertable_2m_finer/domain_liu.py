from proteus import *
from proteus.default_p import *

nd = 2

# BASE case geometry from the reference setup:
# 40 m buffer + 20 m inlet strip + 40 m buffer, with 10 m total depth.
L = (100.0, 50.0)

boundaryTags = {"bottom": 1, "right": 2, "top": 3, "left": 4}

vertices = [
    [0.0,  0.0],   # 0: bottom-left
    [L[0], 0.0],   # 1: bottom-right
    [L[0], L[1]],  # 2: top-right
    [0.0,  L[1]],  # 3: top-left
]
vertexFlags = [
    boundaryTags["bottom"],
    boundaryTags["bottom"],
    boundaryTags["top"],
    boundaryTags["top"],
]
segments = [
    [0, 1],  # bottom
    [1, 2],  # right
    [2, 3],  # top
    [3, 0],  # left
]
segmentFlags = [
    boundaryTags["bottom"],
    boundaryTags["right"],
    boundaryTags["top"],
    boundaryTags["left"],
]
regions = [[L[0] / 2.0, L[1] / 2.0]]
regionFlags = [1]

domain = Domain.PlanarStraightLineGraphDomain(
    vertices=vertices,
    vertexFlags=vertexFlags,
    segments=segments,
    segmentFlags=segmentFlags,
    regions=regions,
    regionFlags=regionFlags,
)
polyfile = "watertable_4m_domain_2d"
domain.writePoly(polyfile)
triangleOptions = "q30Dena0.05"

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexLobattoQuadrature(nd, 1)
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd - 1, 1)
subgridError = None

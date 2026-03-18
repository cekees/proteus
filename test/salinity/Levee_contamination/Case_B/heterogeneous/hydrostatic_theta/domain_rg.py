from proteus import Domain
from proteus import *
from proteus.default_n import *

nd = 2

x_left = 0.0
x_right = 50.0
base = 0.0
ground = 6.25
levee_height = 1.875
crest_width = 6.25
slope = 3.0

L = [x_right - x_left, ground + levee_height, 1.0]

regularGrid = False

boundaries = ['bottom', 'top', 'left', 'right', 'leftTop', 'rightTop']
boundaryTags = dict((key, i + 1) for (i, key) in enumerate(boundaries))

center = 0.5 * (x_left + x_right)
crest_left = center - 0.5 * crest_width
crest_right = center + 0.5 * crest_width
toe_left = crest_left - slope * levee_height
toe_right = crest_right + slope * levee_height

vertices = [
    [x_left, base],
    [x_left, ground],
    [toe_left, ground],
    [crest_left, ground + levee_height],
    [crest_right, ground + levee_height],
    [toe_right, ground],
    [x_right, ground],
    [x_right, base],
]

vertexFlags = [
    boundaryTags['left'],
    boundaryTags['left'],
    boundaryTags['leftTop'],
    boundaryTags['leftTop'],
    boundaryTags['top'],
    boundaryTags['rightTop'],
    boundaryTags['right'],
    boundaryTags['right'],
]

segments = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 0],
    [2, 5],
]

segmentFlags = [
    boundaryTags['left'],
    boundaryTags['leftTop'],
    boundaryTags['leftTop'],
    boundaryTags['top'],
    boundaryTags['rightTop'],
    boundaryTags['rightTop'],
    boundaryTags['right'],
    boundaryTags['bottom'],
    0,
]

regions = [
    [0.1, 0.1],
    [center, ground + 0.25],
]
regionFlags = [1, 2]

domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
                                              vertexFlags=vertexFlags,
                                              segments=segments,
                                              segmentFlags=segmentFlags,
                                              regions=regions,
                                              regionFlags=regionFlags)
domain.regionLegend = {'background': 1,
                       'levee_region': 2,
                       'default': 0}

domain.writePoly('bio2d_N')

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}

elementQuadrature = SimplexLobattoQuadrature(nd, 1)
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd - 1, 1)

nnx = 161
nny = 21
nLevels = 1
triangleFlag = 0
triangleOptions = "pAq30Dena%f" % (0.5 * (L[0] / (nnx - 1)) ** 2,)

subgridError = None

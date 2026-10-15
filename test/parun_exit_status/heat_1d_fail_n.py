"""heat_1d_n with Newton forced to fail on every step.

With no Newton iterations allowed, Newton reports a solver failure before
the first solve, FixedStep cannot recover, and calculateSolution gives up
with "System Step Failed".
"""
from heat_1d_n import *

maxNonlinearIts = 0

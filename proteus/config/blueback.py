# Blueback (NAVY DSRC) is the same machine class as Carpenter and Barfoot:
# HPE Cray EX4000, AMD EPYC 9654 (Genoa), cray-mpich over Slingshot,
# cray-libsci for BLAS/LAPACK, same PUMI/SCOREC link set. Reuse Carpenter's
# configuration rather than duplicating it, so the three cannot drift apart.
from .carpenter import *

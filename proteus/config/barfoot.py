# Barfoot (ERDC DSRC) is the same machine class as Carpenter: HPE Cray EX4000,
# AMD EPYC 9654 (Genoa), cray-mpich over Slingshot, cray-libsci for BLAS/LAPACK,
# and the same PUMI/SCOREC link set. Rather than duplicate that configuration
# and let the two drift apart, reuse Carpenter's -- which also picks up its
# PrgEnv-aware libsci flavour detection.
from .carpenter import *

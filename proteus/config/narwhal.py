# Narwhal (NAVY DSRC) is an HPE Cray EX like Carpenter/Barfoot/Blueback, but an
# earlier generation: AMD EPYC 7H12 (Rome) rather than Genoa, cray-mpich 8.1.x
# rather than 9.1, and an older cray-libsci. The proteus-relevant configuration
# -- cray-libsci for BLAS/LAPACK, cray-mpich, the PUMI/SCOREC link set -- is
# the same, and Carpenter's libsci flavour detection handles the older layout
# (that tree carries both libsci_gnu.so and ABI-suffixed libsci_gnu_82.so).
from .carpenter import *

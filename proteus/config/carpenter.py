from .default import *
import glob as _glob

# The cray-libsci flavour follows the loaded Programming Environment:
# PrgEnv-cray provides libsci_cray, PrgEnv-gnu provides libsci_gnu. This file
# previously hardcoded 'sci_cray', which fails to link under PrgEnv-gnu with
# "cannot find -lsci_cray" -- the GNU libsci tree contains only libsci_gnu*.
# proteus 1.9.0 is verified on GCC and not on Cray clang, so PrgEnv-gnu is the
# supported toolchain here; detect rather than assume.
_libsci_dir = os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR", ""), "lib")
if _glob.glob(os.path.join(_libsci_dir, "libsci_cray.*")):
    _libsci = "sci_cray"
elif _glob.glob(os.path.join(_libsci_dir, "libsci_gnu.*")):
    _libsci = "sci_gnu"
else:
    _libsci = "sci_gnu"

PROTEUS_PRELOAD_LIBS=[]
PROTEUS_EXTRA_LINK_ARGS=['-L'+os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR"),'lib'),'-l'+_libsci] + platform_extra_link_args
PROTEUS_EXTRA_FC_LINK_ARGS=['-L'+os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR"),'lib'),'-l'+_libsci]
PROTEUS_BLAS_LIB_DIR = os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR"),'lib')
PROTEUS_BLAS_LIB   = _libsci
PROTEUS_LAPACK_LIB_DIR = os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR"),'lib')
PROTEUS_LAPACK_LIB = _libsci
PROTEUS_MPI_INCLUDE_DIRS = [os.path.join(os.getenv("CRAY_MPICH_DIR"),'include')]
PROTEUS_MPI_LIB_DIRS = [os.path.join(os.getenv("CRAY_MPICH_DIR"),'lib')]
PROTEUS_MPI_LIBS =[]
#PROTEUS_SUPERLU_LIB_DIR = os.path.join(prefix,'lib64')
PROTEUS_SCOREC_LIBS = [
    'spr',
    'ma',
    'parma',
#    'apf_zoltan',
    'mds',
    'apf',
    'mth',
    'gmi',
    'pcu',
    'lion',
#    'zoltan',
    'parmetis',
    'metis',
    'sam',
    'bz2']+PROTEUS_PETSC_LIBS

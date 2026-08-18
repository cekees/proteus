from .default import *

PROTEUS_PRELOAD_LIBS=[]
PROTEUS_EXTRA_LINK_ARGS=['-L'+os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR"),'lib'),'-lsci_cray'] + platform_extra_link_args
PROTEUS_EXTRA_FC_LINK_ARGS=['-L'+os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR"),'lib'),'-lsci_cray']
PROTEUS_BLAS_LIB_DIR = os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR"),'lib')
PROTEUS_BLAS_LIB   = 'sci_cray'
PROTEUS_LAPACK_LIB_DIR = os.path.join(os.getenv("CRAY_LIBSCI_PREFIX_DIR"),'lib')
PROTEUS_LAPACK_LIB = 'sci_cray'
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

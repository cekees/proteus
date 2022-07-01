from .default import *

PROTEUS_MPI_INCLUDE_DIR = "/usr/local/packages/mvapich2/2.3.7/intel-2021.5.0/include"
PROTEUS_MPI_LIB_DIR = "/usr/local/packages/mvapich2/2.3.7/intel-2021.5.0/lib"
PROTEUS_MPI_INCLUDE_DIRS = [PROTEUS_MPI_INCLUDE_DIR]
PROTEUS_MPI_LIB_DIRS = [PROTEUS_MPI_LIB_DIR]
PROTEUS_SCOREC_LIB_DIRS += PROTEUS_MPI_LIB_DIRS
PROTEUS_SCOREC_INCLUDE_DIRS += PROTEUS_MPI_INCLUDE_DIRS

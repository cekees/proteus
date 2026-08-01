import os
from os.path import join as pjoin
import sys
import platform
import site
PROTEUS_PRELOAD_LIBS=[]
prefix = os.getenv('PROTEUS_PREFIX')
if not prefix:
    prefix = sys.exec_prefix

PROTEUS_OPT = os.getenv('PROTEUS_OPT')
if not PROTEUS_OPT:
    PROTEUS_OPT=[]
else:
    PROTEUS_OPT=PROTEUS_OPT.split()

PROTEUS_INCLUDE_DIR = pjoin(prefix, 'include')
PROTEUS_LIB_DIR = pjoin(prefix, 'lib')

platform_extra_compile_args = []
platform_extra_link_args = []
platform_blas_h = None
platform_lapack_h = None
platform_lapack_integer = None
if sys.platform == 'darwin':
    platform_extra_compile_args = ['-DPETSC_INCLUDE_AS_C', '-DPETSC_SKIP_COMPLEX']
    platform_extra_link_args = ['-L'+PROTEUS_LIB_DIR,'-Wl,-rpath,' + PROTEUS_LIB_DIR]
    platform_blas_h = r'"proteus_blas.h"'
    platform_lapack_h = r'"proteus_lapack.h"'
    major,minor = platform.mac_ver()[0].split('.')[0:2]
    os.environ["MACOSX_DEPLOYMENT_TARGET"]= major+'.'+minor
elif sys.platform.startswith('linux'):
    platform_extra_compile_args = ['-DPETSC_INCLUDE_AS_C', '-DPETSC_SKIP_COMPLEX']
    platform_extra_link_args = ['-L'+PROTEUS_LIB_DIR,'-Wl,-rpath,' + PROTEUS_LIB_DIR]
    platform_blas_h = r'"proteus_blas.h"'
    platform_lapack_h = r'"proteus_lapack.h"'

PROTEUS_EXTRA_COMPILE_ARGS= ['-Wall',
                             '-DF77_POST_UNDERSCORE',
                             '-DUSE_BLAS',
                             '-DCMRVEC_BOUNDS_CHECK',
                             '-DMV_VECTOR_BOUNDS_CHECK'] + platform_extra_compile_args

def get_flags(package):
    """ Checks the environment for presence of PACKAGE_DIR

    And either returns PACKAGE_DIR/[include, lib] or the Proteus include flags.

    This supports building Proteus using packages provides via environment variables.
    """

    package_dir_env = os.getenv(package.upper() + '_DIR')
    if package_dir_env:
        include_dir = pjoin(package_dir_env, 'include')
        lib_dir = pjoin(package_dir_env, 'lib')
    else:
        include_dir = PROTEUS_INCLUDE_DIR
        lib_dir = PROTEUS_LIB_DIR
    return include_dir, lib_dir

def get_petsc_flags():
    """ Like get_flags('petsc'), but with the extra fallbacks PETSc's own
    build layout needs.

    A PETSc source checkout only has real (configured) headers under
    $PETSC_DIR/$PETSC_ARCH/include, not $PETSC_DIR/include directly -- and
    anything PETSc downloaded on your behalf (SuperLU, SuperLU_DIST, ...)
    only ever lands in a separate --prefix install (PROTEUS_INCLUDE_DIR
    below), never under $PETSC_DIR itself, arch subdirectory included.
    get_flags('petsc') alone picks $PETSC_DIR/include whenever $PETSC_DIR is
    set (the common case for any PETSc-based build), silently resolving to
    headers that don't exist. Caught via proteus/superluWrappers.c failing
    to find slu_ddefs.h in a from-scratch sdist build: that extension's
    include path comes straight from this value, unlike most others, which
    also pick up PROTEUS_INCLUDE_DIR and so don't visibly break even when
    this resolves wrong.

    PROTEUS_INCLUDE_DIR (PROTEUS_PREFIX, or sys.exec_prefix if that isn't
    set) is checked first, ahead of $PETSC_DIR-based guesses, precisely
    because it's the one location proven to hold everything a --prefix=
    PETSc build installs, downloaded packages included -- unlike
    $PETSC_DIR/$PETSC_ARCH/include, which can have PETSc's own configured
    headers (petscconf.h) without also having what --download-x packages
    contributed.
    """
    candidates = [(PROTEUS_INCLUDE_DIR, PROTEUS_LIB_DIR)]
    petsc_dir_env = os.getenv('PETSC_DIR')
    if petsc_dir_env:
        candidates.append((pjoin(petsc_dir_env, 'include'), pjoin(petsc_dir_env, 'lib')))
        petsc_arch_env = os.getenv('PETSC_ARCH')
        if petsc_arch_env:
            candidates.append((pjoin(petsc_dir_env, petsc_arch_env, 'include'),
                                pjoin(petsc_dir_env, petsc_arch_env, 'lib')))
    for include_dir, lib_dir in candidates:
        if os.path.isfile(pjoin(include_dir, 'petscconf.h')):
            return include_dir, lib_dir
    # None of the candidates look like a real configured PETSc install;
    # keep the historical behavior (first candidate) rather than guess
    # further, so this fails the same way it always did if something about
    # the environment is genuinely unusual.
    return candidates[0]

PROTEUS_BLAS_INCLUDE_DIR, PROTEUS_BLAS_LIB_DIR = get_flags('blas')
PROTEUS_EXTRA_LINK_ARGS=[]

if sys.platform == 'darwin':
    PROTEUS_BLAS_LIB ='m'
    PROTEUS_BLAS_LIB_DIR = PROTEUS_LIB_DIR
    PROTEUS_BLAS_INCLUDE_DIR = PROTEUS_INCLUDE_DIR
    PROTEUS_EXTRA_LINK_ARGS=platform_extra_link_args
elif sys.platform.startswith('linux'):
    PROTEUS_BLAS_LIB   ='openblas'
    PROTEUS_BLAS_INCLUDE_DIR, PROTEUS_BLAS_LIB_DIR = get_flags('blas')


PROTEUS_CHRONO_INCLUDE_DIR, PROTEUS_CHRONO_LIB_DIR = get_flags('chrono')

PROTEUS_CHRONO_CXX_FLAGS = []
chrono_cmake_file_path = os.path.join(PROTEUS_CHRONO_LIB_DIR,'cmake','Chrono','ChronoConfig.cmake')
if not os.path.isfile(chrono_cmake_file_path):
    chrono_cmake_file_path = os.path.join(PROTEUS_CHRONO_LIB_DIR,'cmake','ChronoConfig.cmake')
    if not os.path.isfile(chrono_cmake_file_path):
        chrono_cmake_file_path = os.path.join(PROTEUS_CHRONO_LIB_DIR,'cmake','Chrono','chrono-config.cmake') 
try:
    with open(chrono_cmake_file_path,'r') as f:
        for l in f:
            if 'set(CHRONO_CXX_FLAGS' in l:
                args = l.split()
                for arg in args:
                    if arg[0] == '-':
                        arg = arg.replace('"', '')
                        arg = arg.replace(')', '')
                        PROTEUS_CHRONO_CXX_FLAGS += [arg]
except FileNotFoundError:
    pass  # chrono not installed; mbd.CouplingFSI extension will be skipped

PROTEUS_EXTRA_FC_COMPILE_ARGS= ['-Wall']
PROTEUS_EXTRA_FC_LINK_ARGS=platform_extra_link_args

PROTEUS_NCURSES_INCLUDE_DIR, PROTEUS_NCURSES_LIB_DIR = get_flags('ncurses')

if platform_blas_h:
    PROTEUS_BLAS_H = platform_blas_h
else:
    PROTEUS_BLAS_H = r'"cblas.h"'

PROTEUS_LAPACK_INCLUDE_DIR, PROTEUS_LAPACK_LIB_DIR = get_flags('lapack')
if platform_lapack_h:
    PROTEUS_LAPACK_H = platform_lapack_h
else:
    PROTEUS_LAPACK_H   = r'"proteus_lapack.h"'

if sys.platform == 'darwin':
    PROTEUS_LAPACK_LIB ='m'
else:
    PROTEUS_LAPACK_LIB = 'openblas'

if platform_lapack_integer:
    PROTEUS_LAPACK_INTEGER = platform_lapack_integer
else:
    PROTEUS_LAPACK_INTEGER = 'int'


PROTEUS_TRIANGLE_INCLUDE_DIR, PROTEUS_TRIANGLE_LIB_DIR = get_flags('triangle')
PROTEUS_TRIANGLE_H = r'"triangle.h"'
PROTEUS_TRIANGLE_LIB ='tri'

PROTEUS_DAETK_INCLUDE_DIR, PROTEUS_DAETK_LIB_DIR = get_flags('daetk')
PROTEUS_DAETK_LIB ='daetk'
PROTEUS_DAETK_LIB_DIRS = [PROTEUS_DAETK_LIB_DIR]

PROTEUS_MPI_INCLUDE_DIR, PROTEUS_MPI_LIB_DIR = get_flags('mpi')
PROTEUS_MPI_INCLUDE_DIRS = [PROTEUS_MPI_INCLUDE_DIR, PROTEUS_MPI_LIB_DIR, os.path.join(PROTEUS_MPI_LIB_DIR,'mpi4py')]+site.getsitepackages()
PROTEUS_MPI_LIB_DIRS = [PROTEUS_MPI_LIB_DIR]
PROTEUS_MPI_LIBS =['mpi']

PROTEUS_PETSC_INCLUDE_DIR, PROTEUS_PETSC_LIB_DIR = get_petsc_flags()
PROTEUS_PETSC_LIB_DIRS = [PROTEUS_PETSC_LIB_DIR]
PROTEUS_PETSC_LIBS = ['petsc']
PROTEUS_PETSC_INCLUDE_DIRS = [PROTEUS_PETSC_INCLUDE_DIR,PROTEUS_PETSC_LIB_DIR]#, os.path.join(PROTEUS_PETSC_LIB_DIR,'petsc4py')]

PROTEUS_SCOREC_INCLUDE_DIR, PROTEUS_SCOREC_LIB_DIR = get_flags('scorec')
PROTEUS_PARMETIS_INCLUDE_DIR, PROTEUS_PARMETIS_LIB_DIR = get_flags('parmetis')
PROTEUS_ZOLTAN_INCLUDE_DIR, PROTEUS_ZOLTAN_LIB_DIR = get_flags('zoltan')
PROTEUS_SCOREC_INCLUDE_DIRS = [PROTEUS_SCOREC_INCLUDE_DIR, PROTEUS_PETSC_INCLUDE_DIR, PROTEUS_ZOLTAN_INCLUDE_DIR, PROTEUS_PARMETIS_INCLUDE_DIR, PROTEUS_MPI_INCLUDE_DIR, PROTEUS_LAPACK_INCLUDE_DIR, PROTEUS_BLAS_INCLUDE_DIR]
PROTEUS_SCOREC_LIB_DIRS =     [PROTEUS_SCOREC_LIB_DIR,     PROTEUS_PETSC_LIB_DIR,     PROTEUS_ZOLTAN_LIB_DIR, PROTEUS_PARMETIS_LIB_DIR,     PROTEUS_MPI_LIB_DIR, PROTEUS_LAPACK_LIB_DIR, PROTEUS_BLAS_LIB_DIR,'/usr/lib/x86_64-linux-gnu/']
PROTEUS_SCOREC_LIBS = [
    'spr',
    'ma',
    'parma',
    'apf_zoltan',
    'mds',
    'apf',
    'mth',
    'gmi',
    'pcu',
    'lion',
    'zoltan',
    'parmetis',
    'metis',
    'sam',
    'bz2']+PROTEUS_PETSC_LIBS

PROTEUS_SCOREC_EXTRA_LINK_ARGS = []
PROTEUS_SCOREC_EXTRA_COMPILE_ARGS = ['-g','-DMESH_INFO']

if os.getenv('SIM_INCLUDE_DIR') is not None:
  PROTEUS_SCOREC_INCLUDE_DIRS = PROTEUS_SCOREC_INCLUDE_DIRS+[pjoin(prefix, 'include'), os.getenv('SIM_INCLUDE_DIR')]
  PROTEUS_SCOREC_LIB_DIRS = PROTEUS_SCOREC_LIB_DIRS+[pjoin(prefix, 'lib'), os.getenv('SIM_LIB_DIR')]
  PROTEUS_SCOREC_LIBS = PROTEUS_SCOREC_LIBS + ['gmi_sim',
    'apf_sim',
    'SimField',
    'SimPartitionedMesh-mpi',
    'SimPartitionWrapper-mpich3',
    'SimMeshing',
    'SimMeshTools',
    'SimModel']

  PROTEUS_SCOREC_EXTRA_COMPILE_ARGS = PROTEUS_SCOREC_EXTRA_COMPILE_ARGS + ['-DPROTEUS_USE_SIMMETRIX']

  PROTEUS_SCOREC_EXTRA_COMPILE_ARGS = PROTEUS_SCOREC_EXTRA_COMPILE_ARGS + ['-DPROTEUS_USE_SIMMETRIX']

PROTEUS_SUPERLU_INCLUDE_DIR = PROTEUS_PETSC_INCLUDE_DIR
PROTEUS_SUPERLU_LIB_DIR = pjoin(prefix, 'lib64')
PROTEUS_SUPERLU_LIB_DIR = pjoin(prefix, 'lib')
PROTEUS_SUPERLU_H   = r'"slu_ddefs.h"'
PROTEUS_SUPERLU_LIB = 'superlu'

PROTEUS_HDF5_INCLUDE_DIR, PROTEUS_HDF5_LIB_DIR = get_flags('hdf5')
PROTEUS_HDF5_LIB_DIRS = [PROTEUS_HDF5_LIB_DIR]
PROTEUS_HDF5_LIBS = ['hdf5']
PROTEUS_HDF5_INCLUDE_DIRS = [PROTEUS_HDF5_INCLUDE_DIR]

if sys.platform.startswith('linux'):
    PROTEUS_EXTRA_LINK_ARGS += ['-Wl,-rpath,' + PROTEUS_HDF5_LIB_DIR]
    PROTEUS_EXTRA_LINK_ARGS += ['-Wl,-rpath,' + PROTEUS_PETSC_LIB_DIR]
    PROTEUS_EXTRA_LINK_ARGS += ['-Wl,-rpath,' + PROTEUS_SCOREC_LIB_DIR]
    PROTEUS_EXTRA_LINK_ARGS += ['-Wl,-rpath,' + PROTEUS_PARMETIS_LIB_DIR]
    PROTEUS_EXTRA_LINK_ARGS += ['-Wl,-rpath,' + PROTEUS_ZOLTAN_LIB_DIR]
    PROTEUS_EXTRA_LINK_ARGS += ['-Wl,-rpath,' + PROTEUS_LAPACK_LIB_DIR]
    PROTEUS_EXTRA_LINK_ARGS += ['-Wl,-rpath,' + PROTEUS_BLAS_LIB_DIR]
    PROTEUS_EXTRA_LINK_ARGS += ['-Wl,-rpath,' + PROTEUS_CHRONO_LIB_DIR]

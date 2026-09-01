#!/usr/bin/env python
""" Test modules for Driven Cavity Stokes preconditioners. """
import proteus.test_utils.TestTools as TestTools
import proteus.LinearAlgebraTools as LAT
import proteus.LinearSolvers as LS
from proteus.iproteus import *
from proteus import defaults

import os
import sys
import inspect
import numpy as np
import h5py
import pickle
import petsc4py
from petsc4py import PETSc
import pytest
import_modules = os.path.join(os.path.dirname(os.path.realpath(__file__)),'import_modules')

TestTools.addSubFolders( inspect.currentframe() )
def create_petsc_vecs(matrix_A):
    """
    Creates a right-hand-side and solution PETSc vector for
    testing ksp solves.

    Parameters
    ----------
    matrix_A: :class:`PETSc.Mat`
        Global matrix object

    Returns
    -------
    vec_lst: tuple
        This is a list of :class:`pypyPETSc.Vec` where the first is
        a vector of ones (usually to act as a RHS-vector) while the
        second vector is a vector of zeros (usually to act as a
        storage vector for the solution).
    """
    b = PETSc.Vec().create()
    x = PETSc.Vec().create()
    b.createWithArray(np.ones(matrix_A.getSizes()[0][0],'d'))
    x.createWithArray(np.zeros(matrix_A.getSizes()[0][0],'d'))
    return (b, x)

@pytest.mark.LinearSolvers
@pytest.mark.modelTest
class TestStokes(proteus.test_utils.TestTools.SimulationTest):
    """Run a Stokes test with mumps LU factorization """

    def setup_method(self):
        stokesDrivenCavity_2d_p = defaults.load_physics('stokesDrivenCavity_2d_p',import_modules)
        stokesDrivenCavity_2d_n = defaults.load_numerics('stokesDrivenCavity_2d_n',import_modules)

    def teardown_method(self):
        """Tear down function. """
        Profiling.closeLog()
        FileList = ['proteus_default.log',
                    'proteus.log',
                    #'rdomain.ele',
                    #'rdomain.edge',
                    #'rdomain.neig',
                    #'rdomain.node',
                    #'rdomain.poly',
                    'drivenCavityStokesTrial.h5',
                    'drivenCavityStokesTrial.xmf']
        self.remove_files(FileList)

    def _setPETSc(self):
        self.nList[0].OptDB.clear()
        for k in self.nList[0].OptDB.getAll(): self.nList[0].OptDB.delValue(k)
        self.nList[0].OptDB.setValue("ksp_type","fgmres")
        self.nList[0].OptDB.setValue("ksp_atol",1e-20)
        self.nList[0].OptDB.setValue("ksp_atol",1e-12)
        self.nList[0].OptDB.setValue("pc_type","fieldsplit")
        self.nList[0].OptDB.setValue("pc_fieldsplit_type","schur")
        self.nList[0].OptDB.setValue("pc_fieldsplit_schur_fact_type","upper")
        self.nList[0].OptDB.setValue("fieldsplit_velocity_ksp_type","preonly")
        self.nList[0].OptDB.setValue("fieldsplit_velocity_pc_type","lu")
        self.nList[0].OptDB.setValue("fieldsplit_pressure_ksp_type","preonly")

    def _setPETSc_LU(self):
        self.nList[0].OptDB.clear()
        for k in self.nList[0].OptDB.getAll(): self.nList[0].OptDB.delValue(k)
        self.nList[0].OptDB.setValue("ksp_type","preonly")
        self.nList[0].OptDB.setValue("pc_type","lu")
        self.nList[0].OptDB.setValue("pc_factor_mat_solver_package","superlu_dist")

    def _runTest(self):
        Profiling.openLog('proteus.log',11)
        Profiling.verbose = True
        self._scriptdir = os.path.dirname(__file__)
        self.ns = NumericalSolution.NS_base(self.so,
                                            self.pList,
                                            self.nList,
                                            self.so.sList,
                                            opts)
        self.ns.calculateSolution('stokes')
        if DUMP_MATRICES:
            _dump_saved_matrix(self.ns)
        actual = h5py.File('drivenCavityStokesTrial.h5','r')
        expected_path = 'comparison_files/' + 'comparison_' + 'drivenCavityStokes' + '_velocity_t1.csv'
        #write comparison file
        #np.array(actual.root.velocity_t1).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
        np.testing.assert_almost_equal(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),np.array(actual['velocity_t1']).flatten(),decimal=2)
        actual.close()

    @pytest.mark.slowTest
    def test_01_FullRun(self):
        stokesDrivenCavity_2d_p = defaults.load_physics('stokesDrivenCavity_2d_p',import_modules)
        stokesDrivenCavity_2d_n = defaults.load_numerics('stokesDrivenCavity_2d_n',import_modules)
        self.pList = [stokesDrivenCavity_2d_p]
        self.nList = [stokesDrivenCavity_2d_n]
        self.nList[0].linearSmoother = proteus.LinearSolvers.Schur_Qp
        self.pList = [stokesDrivenCavity_2d_p]
        self.nList = [stokesDrivenCavity_2d_n]
        defaults.reset_default_so()
        self.so = default_so
        self.so.tnList = [0.,1.]
        self.so.name = self.pList[0].name
        self.so.sList = self.pList[0].name
        self.so.sList = [default_s]
        self._setPETSc()
        self._runTest()
        relpath = 'comparison_files/Qp_expected.log'
        actual_log = TestTools.NumericResults.build_from_proteus_log('proteus.log')
        expected_log = TestTools.NumericResults.build_from_proteus_log(os.path.join(self._scriptdir,
                                                                                    relpath))
        plot_lst = [(1.0,0,0),(1.0,1,0),(1.0,2,0)]
        L1 = expected_log.get_ksp_resid_it_info(plot_lst)
        L2 = actual_log.get_ksp_resid_it_info(plot_lst)
        assert L1 == L2

    @pytest.mark.slowTest
    def test_02_FullRun(self):
        stokesDrivenCavity_2d_p = defaults.load_physics('stokesDrivenCavity_2d_p',import_modules)
        stokesDrivenCavity_2d_n = defaults.load_numerics('stokesDrivenCavity_2d_n',import_modules)
        self.pList = [stokesDrivenCavity_2d_p]
        self.nList = [stokesDrivenCavity_2d_n]
        self.pList = [stokesDrivenCavity_2d_p]
        self.nList = [stokesDrivenCavity_2d_n]
        defaults.reset_default_so()
        self.so = default_so
        self.so.tnList = [0.,1.]
        self.so.name = self.pList[0].name
        self.so.sList = self.pList[0].name
        self.so.sList = [default_s]
        self._setPETSc_LU()
        self._runTest()


def initialize_schur_ksp_obj(matrix_A, schur_approx):
    """
    Creates a right-hand-side and solution PETSc4Py vector for
    testing ksp solves.

    Parameters
    ----------
    matrix_A: :class:`PETSc.Mat`
        Global matrix object.
    schur_approx: :class:`LS.SchurPrecon`

    Returns
    -------
    ksp_obj: :class:`PETSc.KSP`
    """
    ksp_obj = PETSc.KSP().create()
    ksp_obj.setOperators(matrix_A,matrix_A)
    pc = schur_approx.pc
    ksp_obj.setPC(pc)
    ksp_obj.setFromOptions()
    pc.setFromOptions()
    pc.setOperators(matrix_A,matrix_A)
    pc.setUp()
    schur_approx.setUp(ksp_obj)
    ksp_obj.setUp()
    ksp_obj.pc.setUp()
    return ksp_obj

# ---------------------------------------------------------------------------
# Archived solver input. Same problem as in test_nse_RANS2P_step.py: the test
# below asserts an EXACT iteration count (`its == 89`) against a matrix that was
# read by bare relative path from the working directory. Nothing in the suite
# writes that file -- NumericalSolution only emits it as a side effect of
# `Profiling.logLevel > 10` -- so the assertion's input depended on log
# verbosity, test order and cwd, and the test errored outright wherever the file
# was absent ("Cannot locate file: dump_stokes_drivenCavityStokesTrial...").
#
# Unlike the AMG matrices, this one turns out to be completely reproducible:
# the dumps from ten different build pathways (conda-dev x3, published, pip,
# spack x2, and the three PETSc-built HPC prefixes) are BIT-IDENTICAL and all
# yield its=89. That is expected -- the Stokes problem is linear, so its
# Jacobian does not inherit the toolchain sensitivity a nonlinear solve has --
# and it means the exact assertion below is sound once the input is pinned.
#
# Regenerate with:
#     PROTEUS_DUMP_MATRICES=1 pytest test_stokesDC_TH_2D.py -k FullRun
# ---------------------------------------------------------------------------
SAVED_MATRIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'saved_matrices')
SAVED_CAVITY_MATRIX = 'stokes_cavity_par_j.bin'
DUMP_MATRICES = os.environ.get('PROTEUS_DUMP_MATRICES', '') not in ('', '0')


def _dump_saved_matrix(ns):
    """Write this run's parallel Jacobian to saved_matrices/ under its canonical
    name, dropping the ".m" and ".info" sidecars the PETSc viewer leaves."""
    os.makedirs(SAVED_MATRIX_DIR, exist_ok=True)
    prefix = os.path.join(SAVED_MATRIX_DIR, 'stokes_regen_')
    ns.modelList[0].viewJacobian(file_prefix=prefix)
    produced = prefix + 'par_j_1'
    target = os.path.join(SAVED_MATRIX_DIR, SAVED_CAVITY_MATRIX)
    if not os.path.exists(produced):
        raise AssertionError('viewJacobian produced no %s' % produced)
    os.replace(produced, target)
    for sidecar in (produced + '.m', produced + '.info'):
        if os.path.exists(sidecar):
            os.remove(sidecar)


@pytest.fixture()
def load_nse_cavity_matrix(request):
    """Loads the archived driven-cavity matrix. Anchored to this file's
    directory: it must not depend on what an earlier test left in cwd."""
    A = LAT.petsc_load_matrix(os.path.join(SAVED_MATRIX_DIR,
                                           SAVED_CAVITY_MATRIX))
    yield A

@pytest.fixture()
def initialize_petsc_options(request):
    """Initializes schur complement petsc options. """
    petsc_options = PETSc.Options()
    petsc_options.clear()
    for k in petsc_options.getAll(): petsc_options.delValue(k)
    petsc_options.setValue('ksp_type','gmres')
    petsc_options.setValue('ksp_gmres_restart',500)
    petsc_options.setValue('ksp_atol',1e-16)
    petsc_options.setValue('ksp_rtol',1.0e-12)
    petsc_options.setValue('ksp_gmres_modifiedgramschmidt','')
    petsc_options.setValue('pc_type','fieldsplit')
    petsc_options.setValue('pc_fieldsplit_type','schur')
    petsc_options.setValue('pc_fieldsplit_schur_fact_type','upper')
    petsc_options.setValue('pc_fieldsplit_schur_precondition','user')
    petsc_options.setValue('fieldsplit_velocity_ksp_type','preonly')
    petsc_options.setValue('fieldsplit_velocity_pc_type', 'lu')
    petsc_options.setValue('fieldsplit_pressure_ksp_type','preonly')

@pytest.mark.LinearSolvers
def test_Schur_Sp_solve_global_null_space(load_nse_cavity_matrix,
                                          initialize_petsc_options):
    """Tests a KSP solve using the Sp Schur complement approximation.
    For this test, the global matrix has a null space because the
    boundary conditions are pure Dirichlet. """
    mat_A = load_nse_cavity_matrix
    b, x = create_petsc_vecs(mat_A)

    solver_info = LS.ModelInfo('interlaced',
                               3,
                               bdy_null_space=True)
    schur_approx = LS.Schur_Sp(L=mat_A,
                               prefix='',
                               solver_info=solver_info)
    petsc_options = initialize_petsc_options
    ksp_obj = initialize_schur_ksp_obj(mat_A,schur_approx)
    ksp_obj.solve(b,x)

    assert ksp_obj.is_converged == True
    assert ksp_obj.its == 89
    assert ksp_obj.norm < np.linalg.norm(b)*1.0e-9 + 1.0e-16
    assert ksp_obj.reason == 2

if __name__ == '__main__':
    pass

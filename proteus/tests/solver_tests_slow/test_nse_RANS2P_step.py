#!/usr/bin/env python
""" Test modules for Driven Cavity Stokes preconditioners. """
from __future__ import absolute_import

import proteus.test_utils.TestTools as TestTools
from proteus import LinearAlgebraTools as LAT
from proteus import LinearSolvers as LS

from proteus.iproteus import *

import os
import sys
import inspect
import numpy as np
import tables
import pickle
import petsc4py
from petsc4py import PETSc
import pytest

proteus.test_utils.TestTools.addSubFolders( inspect.currentframe() )
try:
    from .import_modules import step2d_so
    from .import_modules import step2d
    from .import_modules import twp_navier_stokes_step2d_p
    from .import_modules import twp_navier_stokes_step2d_n
except:
    from import_modules import step2d_so
    from import_modules import step2d
    from import_modules import twp_navier_stokes_step2d_p
    from import_modules import twp_navier_stokes_step2d_n

def load_simulation(context_options_str=None):
    """
    Loads a two-phase step problem with settings

    Parameters
    ----------
    settings:

    Returns:
    --------

    """
    from proteus import Context
    Profiling.openLog("proteus.log",11)
    Profiling.verbose=True
    Context.contextOptionsString=context_options_str

    reload(step2d_so)
    reload(step2d)
    reload(twp_navier_stokes_step2d_p)
    reload(twp_navier_stokes_step2d_n)

    pList = [twp_navier_stokes_step2d_p]
    nList = [twp_navier_stokes_step2d_n]
    pList[0].name = 'step2d'        
    so = step2d_so
    so.name = pList[0].name
    so.sList = pList[0].name
    so.sList = [default_s]
    _scriptdir = os.path.dirname(__file__)
    Profiling.logAllProcesses = True
    ns = NumericalSolution.NS_base(so,
                                   pList,
                                   nList,
                                   so.sList,
                                   opts)
    return ns

def runTest(ns, name):
    ns.calculateSolution(name)
    actual_log = TestTools.NumericResults.build_from_proteus_log('proteus.log')
    return actual_log

def test_01_FullRun():
    """ Runs two-dimensional step problem with the settings:
        * Strongly enforced Free-slip BC.
        * Pressure Projection Stablization.
        * he = 0.05
    """
    TestTools.SimulationTest._setPETSc(petsc_file = os.path.join(os.path.dirname(__file__),
                                                                 'import_modules/petsc.options.schur'))
    context_options_str='he=0.05'
    ns = load_simulation(context_options_str)
    actual_log = runTest(ns,'test_1')

    L1 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,0)])
    L2 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,1)])
    L3 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,2)])
    L4 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,3)])
    L5 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,4)])
    L6 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,5)])
    
    assert L1[0][1]==25
    assert L2[0][1]==31
    assert L3[0][1]==43
    assert L4[0][1]==39
    assert L5[0][1]==40
    assert L6[0][1]==35

def test_02_FullRun():
    """ Runs two-dimensional step problem with the settings:
        * Strongly enforced no-slip BC.
        * Pressure Projection Stablization.
        * he = 0.05
    """
    TestTools.SimulationTest._setPETSc(petsc_file = os.path.join(os.path.dirname(__file__),
                                                                 'import_modules/petsc.options.schur'))
    context_options_str="boundary_condition_type='ns'"
    ns = load_simulation(context_options_str)
    actual_log = runTest(ns,'test_2')

    L1 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,0)])
    L2 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,1)])
    L3 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,2)])
    L4 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,3)])
    L5 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,4)])
    L6 = actual_log.get_ksp_resid_it_info([(' step2d ',1e+18,0,5)])

    assert L1[0][1]==34
    assert L2[0][1]==38
    assert L3[0][1]==50
    assert L4[0][1]==47
    assert L5[0][1]==48
    assert L6[0][1]==48

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
    b.createWithArray(np.ones(matrix_A.getSizes()[0][0]))
    x.createWithArray(np.zeros(matrix_A.getSizes()[0][0]))
    return (b, x)

def initialize_asm_ksp_obj(matrix_A):
    """
    Creates a right-hand-side and solution PETSc vector for
    testing ksp solves.

    Parameters
    ----------
    matrix_A: :class:`PETSc.Mat`
        Global matrix object.

    Returns
    -------
    ksp_obj: :class:`PETSc.KSP`
    """
    ksp_obj = PETSc.KSP().create()
    ksp_obj.setOperators(matrix_A,matrix_A)
    ksp_obj.setFromOptions()
    ksp_obj.setUp()
    return ksp_obj

def build_amg_index_sets(L_sizes):
    """
    Create PETSc index sets for the velocity components of a saddle
    point matrix

    Parameters
    ----------
    L_sizes :
        Sizes of original saddle-point system

    Returns:
    --------
    Index_Sets : lst
        List of velocity index sets
    """
    neqns = L_sizes[0][0]
    velocityDOF=[]
    for start in range(1,3):
        velocityDOF.append(np.arange(start=start,
                                     stop=1+neqns,
                                     step=3,
                                     dtype='i'))
    velocityDOF_full=np.vstack(velocityDOF).transpose().flatten()
    velocity_u_DOF = []
    velocity_u_DOF.append(np.arange(start=0,
                                    stop=2*neqns//3,
                                    step=2,
                                    dtype='i'))
    velocity_u_DOF_full = np.vstack(velocity_u_DOF).transpose().flatten()
    velocity_v_DOF = []
    velocity_v_DOF.append(np.arange(start=1,
                                    stop=1+2*neqns//3,
                                    step=2,
                                    dtype='i'))
    velocity_v_DOF_full = np.vstack(velocity_v_DOF).transpose().flatten()
    isvelocity = PETSc.IS()
    isvelocity.createGeneral(velocityDOF_full)
    isu = PETSc.IS()
    isu.createGeneral(velocity_u_DOF_full)
    isv = PETSc.IS()
    isv.createGeneral(velocity_v_DOF_full)
    return [isvelocity, isu, isv]

def clear_petsc_options():
    for key in PETSc.Options().getAll():
        PETSc.Options().delValue(key)

def initialize_velocity_block_petsc_options():
    petsc_options = PETSc.Options()
    petsc_options.setValue('ksp_type','gmres')
    petsc_options.setValue('ksp_gmres_restart',100)
#    petsc_options.setValue('ksp_pc_side','right')
    petsc_options.setValue('ksp_atol',1e-8)
    petsc_options.setValue('ksp_gmres_modifiedgramschmidt','')
    petsc_options.setValue('pc_type','hypre')
    petsc_options.setValue('pc_type_hypre_type','boomeramg')

def initialize_velocity_block_petsc_options_2():
    petsc_options = PETSc.Options()
    petsc_options.setValue('ksp_type','gmres')
    petsc_options.setValue('ksp_gmres_restart',100)
    petsc_options.setValue('ksp_atol',1e-8)
    petsc_options.setValue('ksp_gmres_modifiedgramschmidt','')

@pytest.fixture()
def load_saddle_point_matrix_1(request):
    """
    Loads a small example of a drivine cavity matrix for
    testing purposes. (Note: this matrix does not have advection)
    """
    A = LAT.petsc_load_matrix(os.path.join
                              (os.path.dirname(__file__),
                               'dump_test_1_step2d_1e+18par_j_0'))
    yield A

@pytest.fixture()
def load_medium_step_matrix(request):
    """
    Loads a medium sized backwards facing step matrix for studying
    different AMG preconditioners.
    """
    A = LAT.petsc_load_matrix(os.path.join
                              (os.path.dirname(__file__),
                               'dump_test_2_step2d_1e+18par_j_0'))
    yield A

@pytest.mark.amg
def test_amg_iteration_matrix_1(load_saddle_point_matrix_1):
    mat_A = load_saddle_point_matrix_1
    petsc_options = initialize_velocity_block_petsc_options()
    L_sizes = mat_A.getSizes()
    index_sets = build_amg_index_sets(L_sizes)
    F_ksp = initialize_asm_ksp_obj(mat_A.createSubMatrix(index_sets[0],
                                                         index_sets[0]))
    b, x = create_petsc_vecs(mat_A.createSubMatrix(index_sets[0],
                                                   index_sets[0]))
    F_ksp.solve(b,x)
    assert F_ksp.its == 27

    PETSc.Options().setValue('pc_hypre_boomeramg_relax_type_all','sequential-Gauss-Seidel')
    F_ksp = initialize_asm_ksp_obj(mat_A.createSubMatrix(index_sets[0],
                                                         index_sets[0]))
    b, x = create_petsc_vecs(mat_A.createSubMatrix(index_sets[0],
                                                   index_sets[0]))

    F_ksp.solve(b,x)
    assert F_ksp.its == 28

    clear_petsc_options()
    initialize_velocity_block_petsc_options()

    PETSc.Options().setValue('pc_hypre_boomeramg_coarsen_type','PMIS')
    F_ksp = initialize_asm_ksp_obj(mat_A.createSubMatrix(index_sets[0],
                                                         index_sets[0]))
    b, x = create_petsc_vecs(mat_A.createSubMatrix(index_sets[0],
                                                   index_sets[0]))

    F_ksp.solve(b,x)
    assert F_ksp.its == 53

    clear_petsc_options()
    initialize_velocity_block_petsc_options()

    PETSc.Options().setValue('pc_hypre_boomeramg_relax_type_all','sequential-Gauss-Seidel')
    PETSc.Options().setValue('pc_hypre_boomeramg_coarsen_type','PMIS')
    F_ksp = initialize_asm_ksp_obj(mat_A.createSubMatrix(index_sets[0],
                                                         index_sets[0]))
    b, x = create_petsc_vecs(mat_A.createSubMatrix(index_sets[0],
                                                   index_sets[0]))

    F_ksp.solve(b,x)
    assert F_ksp.its == 62

def test_amg_iteration_matrix_2(load_saddle_point_matrix_1):
    mat_A = load_saddle_point_matrix_1
    petsc_options = initialize_velocity_block_petsc_options_2()
    L_sizes = mat_A.getSizes()
    index_sets = build_amg_index_sets(L_sizes)

    F_ksp = initialize_asm_ksp_obj(mat_A.createSubMatrix(index_sets[0],
                                                         index_sets[0]))

    F_ksp.pc.setType('fieldsplit')
    F_ksp.pc.setFieldSplitIS(('v1',index_sets[1]),('v2',index_sets[2]))

    F_ksp.pc.getFieldSplitSubKSP()[0].setType('richardson')
    F_ksp.pc.getFieldSplitSubKSP()[1].setType('richardson')
    F_ksp.pc.getFieldSplitSubKSP()[0].pc.setType('hypre')
    F_ksp.pc.getFieldSplitSubKSP()[0].pc.setHYPREType('boomeramg')
    F_ksp.pc.getFieldSplitSubKSP()[1].pc.setType('hypre')
    F_ksp.pc.getFieldSplitSubKSP()[1].pc.setHYPREType('boomeramg')

    b, x = create_petsc_vecs(mat_A.createSubMatrix(index_sets[0],
                                                   index_sets[0]))
    F_ksp.solve(b,x)
    assert F_ksp.its == 7

if __name__ == '__main__':
    pass

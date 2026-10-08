"""PETSc options given as numerics data, set only while NS_base needs them.

The options database is a process-wide singleton. Options a numerics module
gives as data (``petscOptions``) are set while NS_base builds its solvers and
while it solves, and then the database is put back as it was.
"""
import pytest


def test_numerics_petsc_options_are_set_only_inside_their_scope():
    PETSc = pytest.importorskip("petsc4py.PETSc")
    from proteus.NumericalSolution import _numerics_petsc_options, _petsc_options_scope

    class N(object):
        linear_solver_options_prefix = "model1_"
        petscOptions = {"ksp_type": "gmres", "-pc_type": "lu"}

    options = _numerics_petsc_options([N()])
    assert options == {"model1_ksp_type": "gmres", "model1_pc_type": "lu"}
    database = PETSc.Options()
    database.setValue("model1_pc_type", "jacobi")          # e.g. from parun -P
    try:
        with _petsc_options_scope(options):
            assert database.getString("model1_ksp_type") == "gmres"
            assert database.getString("model1_pc_type") == "lu"
        assert not database.hasName("model1_ksp_type")    # removed
        assert database.getString("model1_pc_type") == "jacobi"   # put back
    finally:
        database.delValue("model1_pc_type")



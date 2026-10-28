"""The build reports which graph partitioners it found, and the report is true.

proteus names no partitioner library itself: PETSc picks the mesh partitioner
and PUMI balances with ParMA. These tests pin that the compiled-in reports
agree with what is on disk, so a build against a PUMI or PETSc without Zoltan,
ParMETIS or METIS says so instead of failing later.
"""
import os
import glob
import pytest


def test_petsc_partitioners_match_petscconf():
    from proteus import cpartitioning
    found = cpartitioning.petscPartitioners()
    assert isinstance(found, list)
    assert set(found) <= {'parmetis', 'chaco', 'party', 'ptscotch'}
    from proteus import config
    confs = [os.path.join(d, 'petscconf.h')
             for d in (config.PROTEUS_PETSC_INCLUDE_DIR, config.PROTEUS_PETSC_LIB_DIR)
             if os.path.exists(os.path.join(d, 'petscconf.h'))]
    if not confs:
        pytest.skip("no petscconf.h to compare against")
    text = open(confs[0]).read()
    for name in ('parmetis', 'chaco', 'party', 'ptscotch'):
        assert (name in found) == (('#define PETSC_HAVE_' + name.upper() + ' 1') in text), name


def test_pumi_partitioners_match_install():
    pumi = pytest.importorskip("proteus.MeshAdaptPUMI.MeshAdapt")
    from proteus import config
    found = pumi.pumiPartitioners()
    assert set(found) == {'zoltan', 'metis'}
    cmake_dir = os.path.join(config.PROTEUS_SCOREC_LIB_DIR, 'cmake', 'SCOREC')
    for name in found:
        tgt = glob.glob(os.path.join(cmake_dir, 'apf_%s-target*.cmake' % name))
        if not tgt:
            assert found[name] is None
            continue
        has = any(('PUMI_HAS_' + name.upper()) in open(f).read() for f in tgt)
        assert found[name] == has, name

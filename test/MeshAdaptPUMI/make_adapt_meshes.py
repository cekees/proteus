"""Regenerate the stored meshes used by test_adaptParallel.py.

Run from this directory, serially, with triangle, tetgen, mpiexec and PUMI's
``split`` utility on PATH::

    python make_adapt_meshes.py

For a unit square (triangle) and a unit cube (tetgen) it writes

    adapt_<name>.dmg            derived geometric model
    adapt_<name>_1p0.smb        serial mesh
    adapt_<name>_2p{0,1}.smb    the same mesh split into 2 parts
    adapt_<name>_4p{0..3}.smb   the same mesh split into 4 parts

so that every rank count adapts the same initial mesh.
"""
import glob
import os
import subprocess
import sys


def reconstruct(name, domain, mesh):
    """Write Reconstructed.{dmg,smb}. PUMI's reconstruction ends the process."""
    from proteus.MeshAdaptPUMI import AdaptHelper, MeshAdapt
    domain.AdaptManager = MeshAdapt.AdaptManager(MeshAdapt.MeshAdapt())
    AdaptHelper.PUMI_helper.reconstructMesh(None, domain, mesh)


def finish(name):
    os.replace('Reconstructed.dmg', 'adapt_%s.dmg' % name)
    os.replace('Reconstructed0.smb', 'adapt_%s_1p0.smb' % name)
    for nparts in (2, 4):
        for f in glob.glob('adapt_%s_%ip*.smb' % (name, nparts)):
            os.remove(f)
        subprocess.check_call(['mpiexec', '-n', str(nparts), 'split',
                               'adapt_%s.dmg' % name, 'adapt_%s_1p.smb' % name,
                               'adapt_%s_%ip.smb' % (name, nparts), str(nparts)])


def square():
    domain = Domain.PlanarStraightLineGraphDomain(
        vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vertexFlags=[1, 2, 3, 4],
        segments=[[0, 1], [1, 2], [2, 3], [3, 0]],
        segmentFlags=[1, 2, 3, 4])
    domain.writePoly('adapt_square')
    subprocess.check_call(['triangle', '-pqa0.005Aen', 'adapt_square.poly'])
    mesh = MeshTools.TriangularMesh()
    mesh.generateFromTriangleFiles('adapt_square.1', base=1)
    reconstruct('square', domain, mesh)


def cube():
    domain = Domain.PiecewiseLinearComplexDomain(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        vertexFlags=[1, 1, 1, 1, 2, 2, 2, 2],
        facets=[[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
                [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]],
        facetFlags=[1, 2, 3, 4, 5, 6])
    domain.writePoly('adapt_cube')
    subprocess.check_call(['tetgen', '-pq1.5a0.002Aen', 'adapt_cube.poly'])
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateFromTetgenFiles('adapt_cube.1', base=1)
    reconstruct('cube', domain, mesh)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        from proteus import Comm, Domain, MeshTools
        Comm.init()
        {'square': square, 'cube': cube}[sys.argv[1]]()
    else:
        for name in ('square', 'cube'):
            subprocess.check_call([sys.executable, __file__, name])
            finish(name)
        for f in glob.glob('adapt_square.*') + glob.glob('adapt_cube.*'):
            if not f.endswith('.dmg'):
                os.remove(f)

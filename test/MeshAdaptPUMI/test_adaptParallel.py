"""PUMI mesh adaptation with the uniform and interface size fields.

Run serially and under ``mpiexec -n 2`` and ``mpiexec -n 4``. Every rank count
loads the same initial mesh, stored pre-split (see make_adapt_meshes.py), so
results are comparable across rank counts. Parallel runs exercise ParMA load
balancing, which MeshAdapt skips on a single rank.
"""
import os

import numpy as np
import pytest
from mpi4py import MPI

from proteus import Comm, Domain, MeshTools, cmeshTools
from proteus.MeshAdaptPUMI import MeshAdapt

testDir = os.path.dirname(os.path.abspath(__file__))
comm = Comm.init()
mpi = MPI.COMM_WORLD

# Initial meshes written by make_adapt_meshes.py
INITIAL = {2: dict(name='square', elements=319),
           3: dict(name='cube', elements=923)}


def setup_adapter(dim, sizeInputs, **properties):
    nparts = comm.size()
    if nparts not in (1, 2, 4):
        pytest.skip('stored meshes are split for 1, 2 and 4 ranks only')
    name = INITIAL[dim]['name']
    adapter = MeshAdapt.MeshAdapt()
    domain = Domain.PUMIDomain(dim=dim, manager=MeshAdapt.AdaptManager(adapter))
    manager = domain.AdaptManager
    manager.sizeInputs = sizeInputs
    manager.adapt = 1
    for key, value in properties.items():
        setattr(manager, key, value)
    model = os.path.join(testDir, 'adapt_%s.dmg' % name)
    mesh = os.path.join(testDir, 'adapt_%s_%ip.smb' % (name, nparts))
    adapter.loadModelAndMesh(model.encode('ascii'), mesh.encode('ascii'))
    adapter.setAdaptProperties(manager)
    return domain


def convert(domain):
    """Convert the PUMI mesh to Proteus; return (global mesh, local mesh)."""
    mesh = MeshTools.TriangularMesh() if domain.nd == 2 else MeshTools.TetrahedralMesh()
    mesh.cmesh = cmeshTools.CMesh()
    mesh.convertFromPUMI(domain, domain.AdaptManager.PUMIAdapter, domain.faceList,
                         domain.regList, parallel=comm.size() > 1, dim=domain.nd)
    local = mesh.subdomainMesh if comm.size() > 1 else mesh
    return mesh, local


def imbalance(local):
    counts = mpi.allgather(local.nElements_global)
    return max(counts) / np.mean(counts)


def edge_lengths(local):
    x = local.nodeArray[local.edgeNodesArray]
    return np.linalg.norm(x[:, 1] - x[:, 0], axis=1)


@pytest.mark.parametrize('dim', [2, 3])
def test_uniform(dim):
    """Uniform refinement splits every edge once: the counts are exact."""
    domain = setup_adapter(dim, [b'uniform'])
    before, local = convert(domain)
    assert before.nElements_global == INITIAL[dim]['elements']
    imbalance_before = imbalance(local)

    domain.AdaptManager.PUMIAdapter.adaptPUMIMesh(b'')
    after, local = convert(domain)

    assert after.nElements_global == 2**dim * before.nElements_global
    assert after.nNodes_global == before.nNodes_global + before.nEdges_global
    assert imbalance(local) <= max(imbalance_before, 1.05) + 0.02


# Serial element counts after one interface adaptation
INTERFACE_ELEMENTS = {2: 2878, 3: 22830}


def circle_distance(nodes, dim):
    center = np.full(dim, 0.5)
    return np.linalg.norm(nodes[:, :dim] - center, axis=1) - 0.3


@pytest.mark.parametrize('dim', [2, 3])
def test_interface(dim):
    """The interface size field refines a band around the zero level set."""
    hmax, hphi = (0.1, 0.025) if dim == 2 else (0.25, 0.08)
    domain = setup_adapter(dim, [b'interface'], hmax=hmax, hmin=hphi / 2,
                           hphi=hphi, numIterations=5)
    adapter = domain.AdaptManager.PUMIAdapter
    before, local = convert(domain)
    # rho, nu, g, dt, dt_next, T and the interface band width (in elements)
    adapter.transferPropertiesToPUMI(np.array([998.2, 1.205]), np.array([1.004e-6, 1.5e-5]),
                                     np.zeros(3), 1.0, 1.0, 1.0, 2.0)
    phi = circle_distance(local.nodeArray, dim)
    adapter.transferFieldToPUMI(b'phi', np.ascontiguousarray(phi.reshape(-1, 1)))

    adapter.adaptPUMIMesh(b'')
    after, local = convert(domain)

    lengths = edge_lengths(local)
    requested = np.zeros((local.nNodes_global, 1))
    adapter.transferFieldToProteus(b'proteus_size', requested)
    ratio = lengths / requested[local.edgeNodesArray, 0].mean(axis=1)
    phi_nodes = circle_distance(local.nodeArray, dim)[local.edgeNodesArray]
    crossing = lengths[phi_nodes[:, 0] * phi_nodes[:, 1] < 0]
    ratio = np.concatenate(mpi.allgather(ratio))
    crossing = np.concatenate(mpi.allgather(crossing))
    conforming = np.mean((ratio > 1 / np.sqrt(2)) & (ratio < np.sqrt(2)))
    print('interface dim=%i ranks=%i' % (dim, comm.size()),
          dict(n=after.nElements_global, imb=round(imbalance(local), 3),
               ratio_median=round(float(np.median(ratio)), 3),
               ratio_p10=round(float(np.percentile(ratio, 10)), 3),
               ratio_p90=round(float(np.percentile(ratio, 90)), 3),
               conforming=round(float(conforming), 3),
               crossing_median_over_hphi=round(float(np.median(crossing)) / hphi, 3),
               crossing_p90_over_hphi=round(float(np.percentile(crossing, 90)) / hphi, 3)))

    # Measured 2026-10-13 (PUMI 4.2.1, ParMA) at 1, 2 and 4 ranks: 2D 2876-2878
    # elements, 90% conforming, crossing edges at 1.00 hphi; 3D 22777-22830
    # elements, 85% conforming, crossing edges at 0.95 hphi; imbalance <= 1.05.
    # The bounds leave room for PUMI version changes, not for regressions.
    assert after.nElements_global == pytest.approx(INTERFACE_ELEMENTS[dim], rel=0.1)
    assert 0.8 <= np.median(ratio) <= 1.2
    assert conforming >= (0.8 if dim == 2 else 0.75)
    assert 0.8 <= np.median(crossing) / hphi <= 1.25
    assert np.percentile(crossing, 90) / hphi <= 1.6
    assert imbalance(local) <= 1.10

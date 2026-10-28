from math import ceil, sqrt, pow
import os
import numpy as np
import numpy.testing as npt
import pytest

from proteus import (Comm,
                     Domain,
                     TransportCoefficients,
                     MeshTools,
                     FemTools,
                     LinearSolvers,
                     Transport,
                     Quadrature,
                     NumericalFlux)
from proteus import default_p as p
from proteus import default_n as n
from importlib import reload
reload(p)
reload(n)

from proteus.Gauges import PointGauges, LineGauges, LineIntegralGauges

from proteus.test_utils.util import setup_profiling, silent_rm

def build1DMesh(p, nnx):
    return MeshTools.MultilevelEdgeMesh(nnx, 1, 1,
                                        p.domain.x[0], 0.0, 0.0,
                                        p.domain.L[0], 1.0, 1.0,
                                        refinementLevels=1,
                                        nLayersOfOverlap=0,
                                        parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)
def build2DMesh(p, nnx, nny):
    return MeshTools.MultilevelTriangularMesh(nnx,nny,1,
                                              p.domain.x[0], p.domain.x[1], 0.0,
                                              p.domain.L[0], p.domain.L[1], 1.0,
                                              refinementLevels=1,
                                              nLayersOfOverlap=0,
                                              parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)

def build3DMesh(p, nnx, nny, nnz):
    return MeshTools.MultilevelTetrahedralMesh(nnx,nny,nnz,
                                               p.domain.x[0], p.domain.x[1], p.domain.x[2],
                                               p.domain.L[0], p.domain.L[1], p.domain.L[2],
                                               refinementLevels=1,
                                               nLayersOfOverlap=0,
                                               parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)

def gauge_setup(nd, total_nodes=None, nnx=None, nny=None, nnz=None, mesh_transform=None, mesh_builder=None):
    comm = Comm.get()

    #Simplified Physics
    p.name="test_gauges"

    p.nd = nd

    class LinearSolution(object):
        def uOfXT(self,x,t):
            return (x[0]+10*x[1]+100*x[2])*(t+1.0)

    p.initialConditions = {0:LinearSolution()}
    p.dirichletConditions = {0: lambda x,flag: None}
    p.domain = Domain.RectangularDomain(name="test_gauges_domain")
    p.coefficients = TransportCoefficients.PoissonEquationCoefficients(aOfX = [lambda x: np.eye(p.nd, p.nd)],
                                                                       fOfX = [lambda x: 0], nc=1, nd=p.nd)
    #Simplified and Incomplete Numerics
    n.femSpaces = {0:FemTools.C0_AffineLinearOnSimplexWithNodalBasis}
    n.elementQuadrature = Quadrature.SimplexGaussQuadrature(p.nd,3)
    n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(p.nd-1,3)
    n.numericalFluxType = NumericalFlux.NoFlux
    n.cfluxtag = None
    n.conservativeFlux = None
    if comm.size() > 1:
        # a direct solve is not available in parallel; nothing is solved here
        n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
        n.levelLinearSolver = LinearSolvers.KSP_petsc4py

    if total_nodes is None:
        total_nodes = 2*comm.size()

    if mesh_builder is not None:
        mlMesh = mesh_builder()
    elif p.nd == 1:
        mlMesh = build1DMesh(p, total_nodes+1)
    elif p.nd == 2:
        if nnx is None:
            nnx = int(ceil(sqrt(total_nodes)))+1
        if nny is None:
            nny = int(ceil(sqrt(total_nodes)))+1
        mlMesh = build2DMesh(p, nnx, nny)
    elif p.nd == 3:
        if nnx is None:
            nnx = int(ceil(pow(total_nodes, 1.0/3.0)))+1
        if nny is None:
            nny = int(ceil(pow(total_nodes, 1.0/3.0)))+1
        if nnz is None:
            nnz = int(ceil(pow(total_nodes, 1.0/3.0)))+1
        mlMesh = build3DMesh(p, nnx, nny, nnz)
    if mesh_transform is not None:
        mesh_transform(mlMesh.meshList[-1].nodeArray)

    model = Transport.MultilevelTransport(p, n, mlMesh)

    return model, p.initialConditions

def run_gauge(p, time_list, nd=3, total_nodes=None, **mesh_options):
    model, initialConditions = gauge_setup(nd, total_nodes, **mesh_options)

    p.attachModel(model, None)

    m = model.levelModelList[-1]
    m.setInitialConditions(initialConditions, time_list[0])
    # behave like a time-stepping model, set tLast to this time
    m.timeIntegration.tLast = time_list[0]
    tCount = 0
    p.calculate()

    for t in time_list[1:]:
        tCount +=1
        m.setInitialConditions(initialConditions, t)
        m.timeIntegration.tLast = t
        p.calculate()

def parse_gauge_output(filename):
    """Read the file on rank 0, after the gauges have written it, and share it."""
    comm = Comm.get().comm.tompi4py()
    comm.barrier()
    result = None
    if comm.rank == 0:
        with open(filename) as f:
            header = f.readline().split(',')
            assert header[0].strip() == 'time'
            gauge_names = [gauge_name.strip() for gauge_name in header[1:]]
            f.seek(0)
            data = np.genfromtxt(f, delimiter=",", skip_header=1)
        result = (gauge_names, data)
    return comm.bcast(result)

def test_2D_point_gauge_output():
    filename = 'test_2D_gauge_output.csv'
    silent_rm(filename)

    p = PointGauges(gauges=((('u0',), ((0, 0, 0), (1, 1, 0))),),
                    fileName=filename)
    time_list=[0.0, 1.0, 2.0]
    run_gauge(p, time_list)

    correct_gauge_names = ['u0 [        0         0         0]', 'u0 [        1         1         0]']
    correct_data = np.asarray([[   0.,    0.,  11.],
                               [   1.,    0.,  22.],
                               [   2.,    0.,  33.]])

    # synchronize processes before attempting to read file

    Comm.get().barrier()

    gauge_names, data = parse_gauge_output(filename)

    assert correct_gauge_names == gauge_names
    npt.assert_allclose(correct_data, data)
    delete_file(filename)

def test_point_gauge_output():
    filename = 'test_gauge_output.csv'
    silent_rm(filename)

    p = PointGauges(gauges=((('u0',), ((0, 0, 0), (1, 1, 1))),),
                    fileName=filename)
    time_list=[0.0, 1.0, 2.0]
    run_gauge(p, time_list)

    correct_gauge_names = ['u0 [        0         0         0]', 'u0 [        1         1         1]']
    correct_data = np.asarray([[   0.,    0.,  111.],
                               [   1.,    0.,  222.],
                               [   2.,    0.,  333.]])

    # synchronize processes before attempting to read file

    Comm.get().barrier()

    gauge_names, data = parse_gauge_output(filename)

    assert correct_gauge_names == gauge_names
    npt.assert_allclose(correct_data, data)
    delete_file(filename)


def test_point_gauge_output_2():
    filename = 'test_gauge_output_2.csv'
    silent_rm(filename)

    p = PointGauges(gauges=((('u0',), ((0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1))),),
                    fileName=filename)
    time_list=[0.0, 1.0, 2.0]
    run_gauge(p, time_list)

    correct_gauge_names = ['u0 [        0         0         0]',
                           'u0 [      0.5       0.5       0.5]',
                           'u0 [        1         1         1]']
    correct_data = np.asarray([[   0.,    0.,  55.5, 111.],
                               [   1.,    0.,  111., 222.],
                               [   2.,    0.,  166.5, 333.]])

    # synchronize processes before attempting to read file
    Comm.get().barrier()

    gauge_names, data = parse_gauge_output(filename)

    assert correct_gauge_names == gauge_names
    npt.assert_allclose(correct_data, data)
    delete_file(filename)

def test_line_integral_gauge_output():
    filename = 'test_line_integral_gauge_output.csv'
    silent_rm(filename)

    lines = (((0, 0, 0), (1, 1, 1)),)
    fields = ('u0', )

    l = LineIntegralGauges(gauges=((fields, lines),),
                   fileName=filename)
    time_list=[0.0, 1.0, 2.0]
    run_gauge(l, time_list)

    correct_gauge_names = ['u0 [        0         0         0] - [        1         1         1]']
    correct_data = np.asarray([[   0., 96.128819820072678],
                               [   1., 192.25763964014536],
                               [   2., 288.38645946021808]])

    # synchronize processes before attempting to read file
    Comm.get().barrier()

    gauge_names, data = parse_gauge_output(filename)
    assert correct_gauge_names == gauge_names
    npt.assert_allclose(correct_data, data)
    delete_file(filename)

def test_2D_line_integral_gauge_output():
    filename = 'test_2D_line_integral_gauge_output.csv'
    silent_rm(filename)

    lines = (((0, 0, 0), (1, 1, 0)),
             ((0, 0, 0), (1, 0, 0)),
             ((0, 0, 0), (0, 1, 0)),
             ((0, 0.5, 0), (1, 0.5, 0)),
             ((0, 1, 0), (1, 1, 0)),
             ((0.5, 0, 0), (0.5, 1, 0)))
    fields = ('u0', )

    l = LineIntegralGauges(gauges=((fields, lines),),
                   fileName=filename)

    time_list=[0.0, 1.0, 2.0, 2.5]
    run_gauge(l, time_list)

    correct_gauge_names = ['u0 [        0         0         0] - [        1         1         0]',
                           'u0 [        0         0         0] - [        1         0         0]',
                           'u0 [        0         0         0] - [        0         1         0]',
                           'u0 [        0       0.5         0] - [        1       0.5         0]',
                           'u0 [        0         1         0] - [        1         1         0]',
                           'u0 [      0.5         0         0] - [      0.5         1         0]']

    correct_data = np.asarray([[0.,   7.77817459,  0.5,   5.,   5.5,  10.5,   5.5],
                               [1.,  15.55634919,  1.,   10.,  11.,   21.,   11.],
                               [2.,  23.33452378,  1.5,  15.,  16.5,  31.5,  16.5],
                               [2.5, 27.22361108,  1.75, 17.5, 19.25, 36.75, 19.25]])

    # synchronize processes before attempting to read file

    Comm.get().barrier()

    gauge_names, data = parse_gauge_output(filename)
    assert correct_gauge_names == gauge_names

    npt.assert_allclose(correct_data, data)
    delete_file(filename)

def test_line_gauge_output():
    filename = 'test_line_output.csv'
    silent_rm(filename)

    lines = (((0, 0, 0), (1, 1, 1)),)
    fields = ('u0', )

    l = LineGauges(gauges=((fields, lines),),
                   fileName=filename)

    time_list=[0.0, 1.0, 2.0]
    # good hard-code for up to 1,024 processes or so, but slow (about 8 seconds on single process).
    run_gauge(l, time_list, total_nodes=2048)
    correct_gauge_names = ['u0 [        0         0         0]', 'u0 [ 0.076923  0.076923  0.076923]',
                           'u0 [  0.15385   0.15385   0.15385]', 'u0 [  0.23077   0.23077   0.23077]',
                           'u0 [  0.30769   0.30769   0.30769]', 'u0 [  0.38462   0.38462   0.38462]',
                           'u0 [  0.46154   0.46154   0.46154]', 'u0 [  0.53846   0.53846   0.53846]',
                           'u0 [  0.61538   0.61538   0.61538]', 'u0 [  0.69231   0.69231   0.69231]',
                           'u0 [  0.76923   0.76923   0.76923]', 'u0 [  0.84615   0.84615   0.84615]',
                           'u0 [  0.92308   0.92308   0.92308]', 'u0 [        1         1         1]']
    correct_data = np.asarray([[0., 0., 8.53846154, 17.07692308, 25.61538462,
                                34.15384615, 42.69230769, 51.23076923, 59.76923077, 68.30769231,
                                76.84615385, 85.38461538, 93.92307692, 102.46153846, 111.],
                               [1., 0., 17.07692308, 34.15384615, 51.23076923,
                                68.30769231, 85.38461538, 102.46153846, 119.53846154, 136.61538462,
                                153.69230769, 170.76923077, 187.84615385, 204.92307692, 222.],
                               [2., 0., 25.61538462, 51.23076923, 76.84615385,
                                102.46153846, 128.07692308, 153.69230769, 179.30769231, 204.92307692,
                                230.53846154, 256.15384615, 281.76923077, 307.38461538, 333.]])

    # synchronize processes before attempting to read file

    Comm.get().barrier()

    gauge_names, data = parse_gauge_output(filename)

    assert correct_gauge_names == gauge_names
    npt.assert_allclose(correct_data, data)
    delete_file(filename)


def test_line_integral_gauge_fine_mesh():
    # erdc#224: elements shorter than 1e-4 of the line length defeated the
    # absolute tolerances in pruneDuplicateSegments
    filename = 'test_line_integral_fine.csv'
    silent_rm(filename)
    lines = (((0, 0.5, 0), (1, 0.5, 0)),)
    l = LineIntegralGauges(gauges=((('u0',), lines),), fileName=filename)
    run_gauge(l, [0.0, 1.0], nd=2, nnx=20001, nny=3)
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    # u = x + 10 y at t = 0, doubled at t = 1: integral over [0,1] at y=.5
    npt.assert_allclose(data[:, 1], [5.5, 11.0], rtol=1e-10)
    delete_file(filename)

def test_point_location_beyond_two_rings():
    if Comm.get().size() > 1:
        pytest.skip("calls getLocalElement directly, which only sees this rank's elements")
    # erdc#426: getLocalElement only searched the star of the nearest node
    # and one ring around it, and fell back to the nearest node's value
    # (wrong, silently) when the containing element was farther out
    model, ic = gauge_setup(2, 100)
    g = PointGauges(gauges=((('u0',), ((0.5, 0.5, 0),)),))
    g.attachModel(model, None)
    femSpace = model.levelModelList[-1].u[0].femSpace
    mesh = femSpace.mesh
    # a node three rings from the point (the old search stopped at two)
    far = int(np.argmin(np.linalg.norm(mesh.nodeArray - np.array([0., 0., 0.]), axis=1)))
    location = (0.37, 0.22, 0.0)
    assert g.getLocalElement(femSpace, location, far) is not None

def test_point_on_a_face_takes_the_lowest_numbered_element():
    # a point on a face between two elements is in both; a discontinuous field
    # has a different value on each side, so the choice must not depend on
    # which node the search started from
    if Comm.get().size() > 1:
        pytest.skip("calls getLocalElement directly, which only sees this rank's elements")
    model, ic = gauge_setup(2, 100)
    g = PointGauges(gauges=((('u0',), ((0.5, 0.5, 0),)),))
    g.attachModel(model, None)
    femSpace = model.levelModelList[-1].u[0].femSpace
    mesh = femSpace.mesh
    checked = 0
    for eb in range(mesh.nElementBoundaries_global):
        if mesh.elementBoundaryElementsArray[eb, 1] < 0:
            continue  # a boundary face
        pair = sorted(int(e) for e in mesh.elementBoundaryElementsArray[eb])
        face = mesh.elementBoundaryNodesArray[eb]
        midpoint = tuple(mesh.nodeArray[face].mean(axis=0))
        for start in face:
            assert g.getLocalElement(femSpace, midpoint, int(start)) == pair[0]
        checked += 1
        if checked == 50:
            break
    assert checked == 50
# erdc#221: bad inputs must fail early with a message that names the problem

def test_point_gauge_accepts_lists():
    filename = 'test_gauge_lists.csv'
    silent_rm(filename)
    g = PointGauges(gauges=((['u0'], [[0, 0, 0], [1, 1, 1]]),),
                    fileName=filename)
    run_gauge(g, [0.0, 1.0])
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    npt.assert_allclose(data, [[0., 0., 111.], [1., 0., 222.]])
    delete_file(filename)

def test_point_gauge_rejects_2d_point():
    with pytest.raises(ValueError, match="3"):
        PointGauges(gauges=((('u0',), ((0.5, 0.5),)),))

def test_line_gauges_reject_2d_endpoints():
    for cls in (LineGauges, LineIntegralGauges):
        with pytest.raises(ValueError, match="3"):
            cls(gauges=((('u0',), (((0, 0), (1, 1)),)),))

def test_line_gauge_rejects_wrong_endpoint_count():
    with pytest.raises(ValueError, match="endpoint"):
        LineGauges(gauges=((('u0',), (((0, 0, 0), (1, 1, 1), (2, 2, 2)),)),))

def test_line_gauge_rejects_zero_length_line():
    with pytest.raises(ValueError, match="zero length"):
        LineIntegralGauges(gauges=((('u0',), (((0, 0, 0), (0, 0, 0)),)),))

def test_gauge_rejects_non_finite_location():
    with pytest.raises(ValueError, match="finite"):
        PointGauges(gauges=((('u0',), ((0, float('nan'), 0),)),))

def test_gauge_bare_string_is_one_field():
    # ('u0') is the string 'u0', which used to be iterated as the fields 'u' and '0'
    g = PointGauges(gauges=((('u0'), ((0, 0, 0),)),))
    assert list(g.fields) == ['u0']
    with pytest.raises(ValueError, match="fields"):
        PointGauges(gauges=((3, ((0, 0, 0),)),))

def test_gauge_rejects_bad_sample_rate_and_active_time():
    pts = ((('u0',), ((0, 0, 0),)),)
    with pytest.raises(ValueError, match="sampleRate"):
        PointGauges(gauges=pts, sampleRate=-1.0)
    with pytest.raises(ValueError, match="activeTime"):
        PointGauges(gauges=pts, activeTime=(2.0, 1.0))
    with pytest.raises(ValueError, match="activeTime"):
        PointGauges(gauges=pts, activeTime=(1.0,))

def test_gauge_unknown_field_names_the_field():
    g = PointGauges(gauges=((('nope',), ((0, 0, 0),)),))
    with pytest.raises(ValueError, match="nope.*u0"):
        run_gauge(g, [0.0])

def count_gauge_rows(filename):
    return len(parse_gauge_output_times(filename))

def parse_gauge_output_times(filename):
    with open(filename) as f:
        f.readline()
        return [float(line.split(',')[0]) for line in f if line.strip()]

def sampled_times(sampleRate, time_list, activeTime=None, filename='test_gauge_sampling.csv'):
    silent_rm(filename)
    g = PointGauges(gauges=((('u0',), ((0.5, 0.5, 0.5),)),),
                    sampleRate=sampleRate, activeTime=activeTime, fileName=filename)
    run_gauge(g, time_list)
    Comm.get().barrier()
    times = parse_gauge_output_times(filename)
    delete_file(filename)
    return times

def test_sample_rate_equal_to_step_is_not_dropped_by_roundoff():
    # erdc#1276: with sampleRate == dt every step must be sampled; i*dt
    # accumulates roundoff that made `time < last_output + sampleRate` skip
    # roughly one step in ten
    dt = 0.1
    time_list = [i*dt for i in range(40)]
    times = sampled_times(dt, time_list)
    assert len(times) == len(time_list)
    npt.assert_allclose(times, time_list)

def test_sample_rate_is_anchored_to_a_grid():
    # erdc#1276: sampling every ceil(sampleRate/dt) steps quantizes the
    # effective rate (dt=0.03, sampleRate=0.1 gave 0.12), so asking for a
    # higher rate had no effect until it crossed a multiple of dt.  The
    # k-th sample must be the first step at or after t0 + k*sampleRate.
    dt, rate = 0.03, 0.1
    time_list = [i*dt for i in range(101)]
    times = np.asarray(sampled_times(rate, time_list))
    expected = [time_list[0]]
    for k in range(1, 30):
        due = k*rate
        expected.append(next(t for t in time_list if t >= due - 1e-9))
    npt.assert_allclose(times[:len(expected)], expected)
    # mean interval matches the requested one to within a step, not 20% off
    assert abs((times[-1] - times[0])/(len(times) - 1) - rate) < dt/10

def test_sample_rate_larger_than_run_gives_single_sample():
    times = sampled_times(10.0, [0.0, 1.0, 2.0])
    assert times == [0.0]

def test_sample_rate_respects_active_time():
    times = sampled_times(0.5, [0.25*i for i in range(21)], activeTime=(1.0, 3.0))
    npt.assert_allclose(times, [1.0, 1.5, 2.0, 2.5, 3.0])


# lines and points that leave the domain (cekees, 2026-10-26): warn and
# restrict to the mesh instead of raising; roundoff-outside is snapped in

def run_line_integral(lines, nd=2, **kw):
    filename = 'test_outside_line_integral.csv'
    silent_rm(filename)
    l = LineIntegralGauges(gauges=((('u0',), lines),), fileName=filename)
    run_gauge(l, [0.0], nd=nd, **kw)
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    delete_file(filename)
    return np.atleast_2d(data)[:, 1:]

def test_line_integral_partly_outside_domain_is_restricted(monkeypatch):
    messages = []
    import proteus.Gauges as G
    monkeypatch.setattr(G, 'logEvent', lambda msg, level=0: messages.append((level, msg)))
    # u = x + 10 y at t=0; inside [0,1]: integral of x + 2.5 is 3
    data = run_line_integral((((-1, 0.25, 0), (2, 0.25, 0)),), total_nodes=100)
    npt.assert_allclose(data, [[3.0]], rtol=1e-10)
    if Comm.get().rank() == 0:  # the segments are pruned on rank 0
        assert any('outside' in msg and level <= 1 for level, msg in messages)

def test_line_gauge_partly_outside_domain_has_points_only_inside():
    filename = 'test_outside_line.csv'
    silent_rm(filename)
    l = LineGauges(gauges=((('u0',), (((-1, 0.25, 0), (2, 0.25, 0)),)),), fileName=filename)
    run_gauge(l, [0.0], nd=2, total_nodes=100)
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    delete_file(filename)
    x = np.array([float(n.split('[')[1].split()[0]) for n in names])
    assert x.min() >= 0 and x.max() <= 1
    npt.assert_allclose(np.atleast_2d(data)[0, 1:], x + 2.5, rtol=1e-10)

def test_line_entirely_outside_domain_is_an_error():
    with pytest.raises(ValueError, match="outside"):
        run_line_integral((((2, 0.25, 0), (3, 0.25, 0)),), total_nodes=100)

def test_line_on_boundary_to_roundoff_is_snapped_2d():
    # "on the bottom" computed slightly differently from the mesh
    data = run_line_integral((((0, -1e-15, 0), (1, -1e-15, 0)),), total_nodes=100)
    npt.assert_allclose(data, [[0.5]], rtol=1e-10)

def test_line_on_boundary_to_roundoff_is_snapped_3d():
    data = run_line_integral((((0, 0.5, -1e-15), (1, 0.5, -1e-15)),), nd=3)
    npt.assert_allclose(data, [[5.5]], rtol=1e-10)

def test_point_on_boundary_to_roundoff_is_snapped():
    # without snapping the point is "off-element" and reads its nearest node (0.3)
    filename = 'test_outside_point.csv'
    silent_rm(filename)
    g = PointGauges(gauges=((('u0',), ((0.34, -1e-15, 0),)),), fileName=filename)
    run_gauge(g, [0.0], nd=2, total_nodes=100)
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    delete_file(filename)
    npt.assert_allclose(np.atleast_2d(data)[0, 1], 0.34, rtol=1e-10)

def test_point_outside_by_more_than_tolerance_still_uses_nearest_node():
    filename = 'test_outside_point2.csv'
    silent_rm(filename)
    g = PointGauges(gauges=((('u0',), ((0.34, -0.5, 0),)),), fileName=filename)
    run_gauge(g, [0.0], nd=2, total_nodes=100)
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    delete_file(filename)
    npt.assert_allclose(np.atleast_2d(data)[0, 1], 0.3, rtol=1e-10)

def delete_file(filename):
    comm = Comm.get().comm.tompi4py()
    comm.barrier()  # every rank has read the file
    if comm.rank == 0 and os.path.exists(filename):
        try:
            os.remove(filename)
        except OSError as e:
            print ("Error: %s - %s" %(e.filename,e.strerror))
        else:
            pass
    

if __name__ == '__main__':
    setup_profiling()
    test_line_gauge_output()
    test_point_gauge_output()
    test_point_gauge_output_2()
    test_line_integral_gauge_output()

def build_turek_cylinder_mesh():
    """The stored 2D mesh of the flow-around-a-cylinder benchmark: a 2.2 x 0.41
    channel with a hole of radius 0.1 centered at (0.2, 0.2)."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'cylinder2D', 'conforming_rans2p', 'mesh_cylinder2D')
    mesh = MeshTools.TriangularMesh()
    mesh.generateFromTriangleFiles(filebase=base, base=1)
    mlMesh = MeshTools.MultilevelTriangularMesh(0, 0, 0, skipInit=True,
                                                nLayersOfOverlap=0,
                                                parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)
    mlMesh.generateFromExistingCoarseMesh(mesh, 1, nLayersOfOverlap=0,
                                          parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)
    return mlMesh

# a line along the channel through the center of the cylinder, with both ends
# a roundoff outside the inflow and outflow boundaries
TUREK_LINE = ((-1e-12, 0.2, 0), (2.2 + 1e-12, 0.2, 0))

def test_line_gauge_across_cylinder_hole(monkeypatch):
    messages = []
    import proteus.Gauges as G
    monkeypatch.setattr(G, 'logEvent', lambda msg, level=0: messages.append((level, msg)))
    filename = 'test_turek_line.csv'
    silent_rm(filename)
    l = LineGauges(gauges=((('u0',), (TUREK_LINE,)),), fileName=filename)
    run_gauge(l, [0.0], nd=2, mesh_builder=build_turek_cylinder_mesh)
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    delete_file(filename)
    x = np.array([float(n.split('[')[1].split()[0]) for n in names])
    # snapped onto the inflow and outflow boundaries
    npt.assert_allclose([x.min(), x.max()], [0.0, 2.2], atol=1e-12)
    # nothing is gauged inside the hole, and points exist on both sides of it
    assert not np.any(np.abs(x - 0.2) < 0.09)
    assert np.any(x < 0.2) and np.any(x > 0.2)
    # the line is gauged up to the hole on the left and from the hole on the right
    npt.assert_allclose([x[x < 0.2].max(), x[x > 0.2].min()], [0.1, 0.3], atol=1e-5)
    # the header prints coordinates to six digits
    npt.assert_allclose(np.atleast_2d(data)[0, 1:], x + 2.0, atol=1e-4)
    if Comm.get().rank() == 0:  # the segments are pruned on rank 0
        assert any('outside' in msg and level <= 1 for level, msg in messages)

def test_line_integral_across_cylinder_hole():
    filename = 'test_turek_line_integral.csv'
    silent_rm(filename)
    lg = 'test_turek_line_points.csv'
    silent_rm(lg)
    run_gauge(LineGauges(gauges=((('u0',), (TUREK_LINE,)),), fileName=lg),
              [0.0], nd=2, mesh_builder=build_turek_cylinder_mesh)
    Comm.get().barrier()
    names, _ = parse_gauge_output(lg)
    delete_file(lg)
    x = np.sort([float(n.split('[')[1].split()[0]) for n in names])
    # the hole is the one gap between consecutive gauge points that is not a mesh edge crossing
    gap = np.argmax(np.diff(x))
    xa, xb = x[gap], x[gap + 1]
    assert xb - xa > 0.15
    exact = (2.2**2/2 + 2*2.2) - ((xb**2 - xa**2)/2 + 2*(xb - xa))
    l = LineIntegralGauges(gauges=((('u0',), (TUREK_LINE,)),), fileName=filename)
    run_gauge(l, [0.0], nd=2, mesh_builder=build_turek_cylinder_mesh)
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    delete_file(filename)
    # xa and xb come from a header that prints six digits
    npt.assert_allclose(np.atleast_2d(data)[0, 1], exact, rtol=1e-5)

def test_gauge_points_on_element_faces_read_the_field_not_a_node():
    # the ends of a line gauge's segments lie on element faces, and roundoff can put
    # them outside both neighbours; they must still be interpolated, not read at the
    # nearest node.  u = x + 10 y, and the line ends on the top boundary at y = 1
    line = (((0.31, -0.5, 0), (0.31, 1.5, 0)),)
    npt.assert_allclose(run_line_integral(line, total_nodes=100), [[0.31 + 5.0]], rtol=1e-12)
    filename = 'test_face_points.csv'
    silent_rm(filename)
    run_gauge(LineGauges(gauges=((('u0',), line),), fileName=filename), [0.0], nd=2, total_nodes=100)
    names, data = parse_gauge_output(filename)
    delete_file(filename)
    y = np.array([float(n.split('[')[1].split()[1]) for n in names])
    npt.assert_allclose(np.atleast_2d(data)[0, 1:], 0.31 + 10*y, rtol=1e-6)

def test_first_gauge_row_is_the_initial_condition(tmp_path, monkeypatch):
    # NumericalSolution used to sample auxiliary variables before the initial
    # conditions were set, so the t0 row of every gauge file was zeros
    from proteus import ADRProblem, NumericalSolution, SplitOperator, defaults, default_s, runtime
    monkeypatch.chdir(tmp_path)
    problem = {
        "adr": {"dim": 2, "components": ["u"],
                "equations": [{
                    "component": "u",
                    "mass": {"m": "u", "dm": {"u": "1"}, "depends_on": {"u": "linear"}},
                    "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                        "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                    "reaction": {"r": "4.0 + 4.0*t - x**2 - y**2", "dr": {}, "depends_on": {}}}]},
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "time": [0.0, 1.0],
        "has_mass": True,
        "dirichlet": [{"component": "u", "region": "all", "where": "x <= tol or y <= tol or y >= 1.0 - tol or x >= 1.0 - tol",
                       "value": "(x**2 + y**2)*(1.0 + t)"}],
        "initial": {"u": "x**2 + y**2"},
        "exact": {"u": "(x**2 + y**2)*(1.0 + t)"},
    }
    runtime.start(log=None)
    problem, pp, nn, system = ADRProblem.setup(problem, {"u": ("CG", 1)}, 8, time={"dt": 0.25})
    ADRProblem.write_poly(pp)
    nn.auxiliaryVariables = [PointGauges(gauges=((('u',), ((0.5, 0.5, 0.0),)),), fileName='initial_row.csv')]
    so = defaults.System_base(name="adr", pnList=[(pp, nn)], sList=[default_s])
    so.tnList = system["tnList"]
    so.systemStepControllerType = SplitOperator.Sequential_FixedStep
    so.dt_system_fixed = system["dt_system_fixed"]
    ns = NumericalSolution.NS_base(so, [pp], [nn], so.sList, runtime.RunOptions())
    ns.calculateSolution("adr")
    names, data = parse_gauge_output('initial_row.csv')
    data = np.atleast_2d(data)
    npt.assert_allclose(data[0], [0.0, 0.5], atol=1e-12)
    npt.assert_allclose(data[-1, 1], 0.5*(1 + data[-1, 0]), rtol=1e-2)

@pytest.mark.parametrize("nd", [2, 3])
def test_bounding_box_prefilter_changes_no_clipping_result(nd):
    # getMeshIntersections only clips the elements a line can touch; the
    # result must equal clipping every element
    import time
    rng = np.random.default_rng(7)
    if nd == 2:
        mesh = MeshTools.MultilevelTriangularMesh(30, 30, 1, 0, 0, 0, 1, 1, 1, refinementLevels=1).meshList[0]
        toPolyhedron = MeshTools.triangleVerticesToNormals
    else:
        mesh = MeshTools.MultilevelTetrahedralMesh(8, 8, 8, 0, 0, 0, 1, 1, 1, refinementLevels=1).meshList[0]
        toPolyhedron = MeshTools.tetrahedronVerticesToNormals
    def brute(endpoints):
        found = set()
        for element in mesh.elementNodesArray:
            clipped = MeshTools.intersectPolyhedron(endpoints, toPolyhedron(mesh.nodeArray[element]))
            if clipped and not np.array_equal(clipped[0], clipped[1]):
                found.add((tuple(clipped[0]), tuple(clipped[1])))
        return found
    lines = [((0.0, 0.5, 0.5 if nd == 3 else 0.0), (1.0, 0.5, 0.5 if nd == 3 else 0.0)),  # along a mesh line
             ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0 if nd == 3 else 0.0)),                       # through vertices
             ((-0.5, 0.3, 0.0), (1.5, 0.3, 0.0))]                                         # leaving the mesh
    for _ in range(10):
        p, q = rng.uniform(-0.2, 1.2, (2, 3))
        if nd == 2:
            p[2] = q[2] = 0.0
        lines.append((tuple(p), tuple(q)))
    for endpoints in lines:
        assert MeshTools.getMeshIntersections(mesh, toPolyhedron, endpoints) == brute(endpoints)

def test_point_a_roundoff_outside_a_curved_boundary_is_interpolated():
    # the middle of an edge on the cylinder's circle, moved outward by roundoff:
    # outside every element, but not on a face aligned with the bounding box.
    # u = x + 10 y, so the interpolated value is exact; a nearest-node read is not
    filename = 'test_curved_boundary_point.csv'
    silent_rm(filename)
    mesh = build_turek_cylinder_mesh().meshList[0]
    cylinder = [eb for eb in range(mesh.nExteriorElementBoundaries_global)
                if np.linalg.norm(mesh.nodeArray[mesh.elementBoundaryNodesArray[mesh.exteriorElementBoundariesArray[eb]]].mean(axis=0)[:2]
                                  - [0.2, 0.2]) < 0.15]
    face = mesh.elementBoundaryNodesArray[mesh.exteriorElementBoundariesArray[cylinder[0]]]
    middle = mesh.nodeArray[face].mean(axis=0)
    offMesh = np.array([0.2, 0.2, 0.0]) - middle   # toward the hole's center
    # the face's own normal, not the radius, is what leaves the mesh by roundoff
    edge = mesh.nodeArray[face[1]] - mesh.nodeArray[face[0]]
    normal = np.array([edge[1], -edge[0], 0.0])/np.linalg.norm(edge)
    point = middle + 1e-9*(normal if normal.dot(offMesh) > 0 else -normal)
    run_gauge(PointGauges(gauges=((('u0',), (tuple(point),)),), fileName=filename),
              [0.0], nd=2, mesh_builder=build_turek_cylinder_mesh)
    Comm.get().barrier()
    names, data = parse_gauge_output(filename)
    delete_file(filename)
    npt.assert_allclose(np.atleast_2d(data)[0, 1], point[0] + 10*point[1], rtol=1e-6)

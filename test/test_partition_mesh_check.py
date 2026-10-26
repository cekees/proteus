"""n and the domain's mesh options must agree about overlap on several ranks."""
import types

import pytest

from proteus import Comm, Domain, MeshTools
from proteus.MeshTools import MeshParallelPartitioningTypes as Types


class _Comm:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _pn(mesh_type, mesh_layers, n_type, n_layers):
    domain = Domain.RectangularDomain(L=(1.0, 1.0))
    options = domain.MeshOptions
    options.parallelPartitioningType = mesh_type
    options.nLayersOfOverlapForParallel = mesh_layers
    p = types.SimpleNamespace(domain=domain, name="model")
    n = types.SimpleNamespace(parallelPartitioningType=n_type, nLayersOfOverlapForParallel=n_layers)
    return p, n


@pytest.mark.parametrize("mesh,n", [
    ((Types.node, 1), (Types.node, 1)),
    ((Types.node, 1), (Types.node, 0)),         # node is full overlap whatever the layers say
    ((Types.element, 0), (Types.element, 0)),
    ((Types.element, 1), (Types.node, 1)),
    ((Types.element, 1), (Types.element, 2))])
def test_agreement_passes(mesh, n, monkeypatch):
    monkeypatch.setattr(Comm, "get", lambda: _Comm(2))
    MeshTools.setDomainFromPhysics(*_pn(*mesh, *n))


@pytest.mark.parametrize("mesh,n", [
    ((Types.node, 1), (Types.element, 0)),      # test/periodic/duct before the fix
    ((Types.element, 0), (Types.node, 1)),      # the VOF reproduction before it set its mesh options
    ((Types.element, 0), (Types.element, 1))])
def test_disagreement_is_an_error_on_two_ranks(mesh, n, monkeypatch):
    monkeypatch.setattr(Comm, "get", lambda: _Comm(2))
    with pytest.raises(ValueError, match="disagree about whether the residual is assembled"):
        MeshTools.setDomainFromPhysics(*_pn(*mesh, *n))


def test_serial_never_complains(monkeypatch):
    monkeypatch.setattr(Comm, "get", lambda: _Comm(1))
    MeshTools.setDomainFromPhysics(*_pn(Types.node, 1, Types.element, 0))

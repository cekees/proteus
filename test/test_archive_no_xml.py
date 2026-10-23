"""proteus builds its archive as ymf dicts and writes no XML.

C2 of the campaign roster: only ``ymf2xmf`` should need XDMF. The archive
(``Archiver.py``), the mesh writers (``MeshTools.py``), the finite-element
writers (``FemTools.py``), the models that drive them and the auxiliary
variables build ymf grid dicts through the archive's builders; the ``.xmf``
a viewer wants is derived from the ``.ymf`` on demand. These tests pin that
no module on the archive path imports an XML library, and that the running
archive is one step of dicts, not a tree.

The one writer that still builds XDMF, ``mbd/CouplingFSI.pyx``, keeps its
own files for the bodies it moves and is outside this check.
"""
import ast
import os

import numpy as np
import pytest

PROTEUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proteus")

ARCHIVE_PATH = ["Archiver.py", "MeshTools.py", "FemTools.py", "AuxiliaryVariables.py",
                "NumericalSolution.py", "Transport.py", "SplitOperator.py"]


def imports_of(path):
    tree = ast.parse(open(path).read(), path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("module", ARCHIVE_PATH)
def test_the_archive_path_imports_no_xml_library(module):
    names = imports_of(os.path.join(PROTEUS, module))
    assert not [n for n in names if n == "xml" or n.startswith("xml.")], module


def test_no_module_of_the_package_builds_xdmf_elements():
    """``SubElement`` was the signature of the hand-built XDMF; none is left."""
    offenders = []
    for root, _, files in os.walk(PROTEUS):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                with open(path) as f:
                    if "SubElement(" in f.read():
                        offenders.append(os.path.relpath(path, PROTEUS))
    assert offenders == []


def test_the_running_archive_is_one_step_of_dicts():
    from proteus.Archiver import ArchiveCollection, ArchiveDomain

    domain = ArchiveDomain()
    mesh = domain.collection("Mesh Spatial_Domain")
    space = domain.collection("Mesh_c0p2_Lagrange")
    assert domain.collection("Mesh Spatial_Domain") is mesh      # found, not duplicated
    assert domain.collection("Mesh_dgp1", create=False) is None   # a continuing run expects it
    assert domain.names() == ["Mesh Spatial_Domain", "Mesh_c0p2_Lagrange"]
    assert domain[0] is mesh and len(domain) == 2
    g = mesh.new_grid(0.25, 1)
    assert mesh.is_current(0.25) and not space.is_current(0.25)
    domain.clear_step()
    assert mesh.grid is None and not mesh.is_current(0.25)
    assert isinstance(g, dict) and g["attributes"] == []


def test_a_grid_scalar_is_inlined_in_the_document():
    """AuxiliaryVariables.writeScalarXdmf used to put the value in a DataItem's text."""
    from proteus.Archiver import ArchiveDomain
    from proteus.AuxiliaryVariables import AV_base

    class Archive(object):
        domain = ArchiveDomain()

    av = AV_base()
    av.ar = Archive()
    av.ar.domain.collection("Mesh Spatial_Domain").new_grid(0.0, 0)
    av.writeScalarXdmf(3.5, "force")
    (attr,) = av.ar.domain[0].grid["attributes"]
    assert attr["name"] == "force" and attr["center"] == "Grid"
    assert attr["data_item"]["values"] == [3.5]
    assert attr["data_item"]["format"] == "XML"

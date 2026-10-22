"""A module that writes ``proteus.X.Y`` imports ``proteus.X`` itself.

``proteus.X`` is attribute access on the package, not an import. It resolves
only because importing ``proteus.X`` anywhere in the process sets ``X`` as an
attribute of ``proteus``. A module that relies on that works for exactly as
long as something else is imported first, and raises ``AttributeError`` the
day someone imports it on its own -- which is how several ``mprans`` modules
came to be unimportable except in a particular order.

The rule used to be a comment repeated in the six files where somebody had
been bitten by it. Six copies of a convention do not stop the seventh file
from getting it wrong; this does. When it was first run it found **23**
offending files, not six.
"""
import ast
import os

import pytest

import proteus

PACKAGE = os.path.dirname(os.path.abspath(proteus.__file__))


def _submodules():
    """Names that could be imported as ``proteus.<name>``."""
    names = set()
    for entry in os.listdir(PACKAGE):
        path = os.path.join(PACKAGE, entry)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "__init__.py")):
            names.add(entry)
        elif entry.endswith((".py", ".so", ".pyx", ".pyd")) and entry != "__init__.py":
            names.add(entry.split(".")[0])
    return names


def _python_files():
    for root, dirs, files in os.walk(PACKAGE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


SUBMODULES = _submodules()
FILES = sorted(_python_files())


def _offences(path):
    """Submodules this file reaches through ``proteus.`` without importing."""
    with open(path, encoding="utf-8", errors="replace") as f:
        source = f.read()
    try:
        tree = ast.parse(source, path)
    except SyntaxError:                      # not ours to police
        return set()
    imported = {alias.name.split(".")[1]
                for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names if alias.name.startswith("proteus.")}
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "proteus"
            and node.attr in SUBMODULES}
    return used - imported


@pytest.mark.parametrize("path", FILES, ids=[os.path.relpath(p, PACKAGE) for p in FILES])
def test_attribute_access_has_a_matching_import(path):
    missing = _offences(path)
    assert not missing, (
        "%s uses %s but does not import %s. proteus.X is attribute access: it "
        "works only while something else has already imported X, so this file "
        "is not importable on its own. Add the import here."
        % (os.path.relpath(path, PACKAGE),
           ", ".join("proteus.%s" % m for m in sorted(missing)),
           ", ".join("proteus.%s" % m for m in sorted(missing))))


def test_the_check_can_fail(tmp_path):
    """The audit is only worth having if it catches the thing it describes."""
    offender = tmp_path / "offender.py"
    offender.write_text("import proteus\n"
                        "x = proteus.Transport.OneLevelTransport\n", encoding="utf-8")
    assert _offences(str(offender)) == {"Transport"}

    fixed = tmp_path / "fixed.py"
    fixed.write_text("import proteus\nimport proteus.Transport\n"
                     "x = proteus.Transport.OneLevelTransport\n", encoding="utf-8")
    assert _offences(str(fixed)) == set()


def test_all_has_no_duplicates():
    assert len(proteus.__all__) == len(set(proteus.__all__)), \
        "duplicate names in proteus.__all__"

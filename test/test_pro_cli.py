"""The ``pro`` command line (proteus.cli) and proteus.runtime.

The end-to-end test runs one model with ``parun`` and with ``pro run`` and
requires the two archives to be bitwise identical: ``pro run`` is meant to
reproduce parun's behaviour while having its own command line.
"""
import math
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from proteus import runtime
from proteus.cli import build_parser, missing
from proteus.cli.run import model_kind

HERE = os.path.dirname(os.path.abspath(__file__))


def pro(*args, cwd=None):
    """Run ``python -m proteus.cli`` (no installed ``pro`` needed)."""
    return subprocess.run([sys.executable, "-m", "proteus.cli"] + list(args),
                          cwd=cwd, capture_output=True, text=True)


def test_help_does_not_start_mpi_or_petsc():
    code = ("import sys; from proteus.cli import build_parser; "
            "build_parser().format_help(); "
            "print('petsc4py' in sys.modules, 'mpi4py' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_help_lists_the_commands():
    out = pro("--help")
    assert out.returncode == 0
    for command in ("run", "spec", "weak", "disc"):
        assert command in out.stdout


@pytest.mark.skipif(sys.version_info < (3, 14),
                    reason="argparse suggests commands from Python 3.14")
def test_typo_suggests_the_command():
    out = pro("rnu")
    assert out.returncode == 2
    assert "maybe you meant 'run'?" in out.stderr


def test_frontend_command_explains_itself():
    out = pro("weak")
    assert out.returncode == 2
    if missing("ymf.symbolic", "sympy"):
        assert "ymf[symbolic]" in out.stderr
    else:
        assert "not implemented yet" in out.stderr


def test_missing():
    assert missing("os", "no_such_module_here", "no_such_pkg.sub") == [
        "no_such_module_here", "no_such_pkg.sub"]


@pytest.mark.parametrize("inputs, message", [
    (["a.py", "b.py", "c.py"], "give one model module"),
    (["model.txt"], "not a .py file"),
    (["no_such_model.py"], "no such file"),
])
def test_run_rejects_bad_inputs(tmp_path, inputs, message):
    out = pro("run", *inputs, cwd=tmp_path)
    assert out.returncode == 2
    assert message in out.stderr


def test_model_kind(tmp_path):
    so = tmp_path / "m_so.py"
    so.write_text("from proteus.default_so import *\npnList = [('p', 'n')]\n")
    case = tmp_path / "case.py"
    case.write_text("# a TwoPhaseFlow case\nmyTpFlowProblem = None\n")
    assert model_kind(str(so)) == "so"
    assert model_kind(str(case)) == "case"


def test_petsc_argv(tmp_path):
    f = tmp_path / "petsc.options"
    f.write_text(textwrap.dedent("""\
        -ksp_type gmres   # the solver
        # a whole-line comment
        -pc_type asm
        """))
    assert runtime.petsc_argv("-pc_type lu", str(f), program="prog") == [
        "prog", "-ksp_type", "gmres", "-pc_type", "asm", "-pc_type", "lu"]
    assert runtime.petsc_argv(["-a", "1"], program="prog") == ["prog", "-a", "1"]
    assert runtime.petsc_argv(program="prog") == ["prog"]


def test_petsc_argv_program_is_never_an_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["-c"])
    assert runtime.petsc_argv() == ["proteus"]
    monkeypatch.setattr(sys, "argv", [])
    assert runtime.petsc_argv() == ["proteus"]


def test_run_options_match_parun_defaults():
    # iproteus.opts copies parun's parser, except that it never declared
    # -t/--hotStartTime (parun's default: infinity), which NS_base reads on
    # a hot start; test_HotStart_rans3p sets it by hand for that reason.
    from proteus.iproteus import opts
    defaults = runtime.RunOptions()
    assert math.isinf(defaults.hotStartTime)
    assert not hasattr(opts, "hotStartTime")
    for name in vars(defaults):
        if name == "hotStartTime":
            continue
        expected = getattr(opts, name)
        if name == "hotStart":      # parun's default is None, i.e. off
            expected = bool(expected)
        assert getattr(defaults, name) == expected, name
    assert runtime.RunOptions() is not runtime.RunOptions()


def _datasets(path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: out.__setitem__(name, np.asarray(obj[()]))
                     if isinstance(obj, h5py.Dataset) else None)
    return out


@pytest.mark.skipif(shutil.which("parun") is None, reason="parun not on PATH")
def test_run_matches_parun_bitwise(tmp_path):
    import numpy as np
    sources = [os.path.join(HERE, "ci", f)
               for f in ("adr.py", "ladr_2d_p.py", "ladr_2d_n.py")]
    for name in ("parun", "pro"):
        (tmp_path / name).mkdir()
        for s in sources:
            shutil.copy(s, tmp_path / name)
    ran = subprocess.run(["parun", "ladr_2d_p.py", "ladr_2d_n.py"],
                         cwd=tmp_path / "parun", capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr[-2000:]
    ran = pro("run", "ladr_2d_p.py", "ladr_2d_n.py", cwd=tmp_path / "pro")
    assert ran.returncode == 0, ran.stderr[-2000:]

    expected = _datasets(tmp_path / "parun" / "ladr_2d.h5")
    actual = _datasets(tmp_path / "pro" / "ladr_2d.h5")
    assert sorted(actual) == sorted(expected)
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name
    assert ((tmp_path / "pro" / "ladr_2d.ymf").read_bytes()
            == (tmp_path / "parun" / "ladr_2d.ymf").read_bytes())

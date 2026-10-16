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


TETGEN_MODEL = ["poisson_3d_tetgen_p.py", "poisson_3d_tetgen_c0p1_n.py"]


def _copy_ci_models(directory):
    directory.mkdir()
    for name in os.listdir(os.path.join(HERE, "ci")):
        if name.endswith(".py"):
            shutil.copy(os.path.join(HERE, "ci", name), directory)


def test_mesh_files_plan():
    from proteus import Domain, MeshTools
    box = Domain.RectangularDomain(L=[1.0, 1.0])
    assert MeshTools.meshFiles(box, MeshTools.getMeshOptions(box)) is None
    plc = Domain.PiecewiseLinearComplexDomain(name="plc")
    plc.polyfile = "plc"
    options = MeshTools.getMeshOptions(plc)
    options.triangleOptions = "VApq1.35q12feena0.01"
    plan = MeshTools.meshFiles(plc, options)
    assert plan["generator"] == "tetgen"
    assert plan["command"] == "tetgen -VApq1.35q12feena0.01 plc.poly"
    assert plan["inputs"] == ["plc.poly"]
    assert plan["outputs"] == ["plc.ele", "plc.node", "plc.face"]


@pytest.mark.skipif(shutil.which("tetgen") is None, reason="tetgen not on PATH")
def test_msh_then_run_without_meshing(tmp_path):
    work = tmp_path / "pro"
    _copy_ci_models(work)
    made = pro("msh", *TETGEN_MODEL, "-l", "0", cwd=work)
    assert made.returncode == 0, made.stderr[-2000:]
    assert "running tetgen" in made.stdout
    assert (work / "meshNoVessel.mesh.ymf").exists()

    again = pro("msh", *TETGEN_MODEL, "-l", "0", cwd=work)
    assert again.returncode == 0 and "up to date" in again.stdout

    # the run reads the mesh pro msh made, and refuses to remesh it
    ran = pro("run", *TETGEN_MODEL, "--no-mesh", cwd=work)
    assert ran.returncode == 0, ran.stderr[-2000:]

    # a context option that changes tetgen's options makes the mesh stale
    stale = pro("run", *TETGEN_MODEL, "--no-mesh", "-C", "Refinement=1", cwd=work)
    assert stale.returncode == 2
    assert "pro msh poisson_3d_tetgen_p.py poisson_3d_tetgen_c0p1_n.py " \
           "-C Refinement=1" in stale.stderr

    if shutil.which("parun") is None:
        return
    reference = tmp_path / "parun"
    _copy_ci_models(reference)
    ran = subprocess.run(["parun"] + TETGEN_MODEL, cwd=reference,
                         capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr[-2000:]
    import numpy as np
    expected = _datasets(reference / "poisson_tetgen_0.h5")
    actual = _datasets(work / "poisson_tetgen_0.h5")
    assert sorted(actual) == sorted(expected)
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name


@pytest.mark.skipif(shutil.which("tetgen") is None, reason="tetgen not on PATH")
def test_edited_mesh_file_is_stale(tmp_path):
    from proteus.cli import mesh
    work = tmp_path / "pro"
    _copy_ci_models(work)
    assert pro("msh", *TETGEN_MODEL, "-l", "0", cwd=work).returncode == 0
    plan = {"generator": "tetgen", "prefix": str(work / "meshNoVessel")}
    import ymf.archive
    record = ymf.archive.read_document(str(work / "meshNoVessel.mesh.ymf"))["mesh"]
    plan["command"] = record["command"]
    plan["inputs"] = [str(work / "meshNoVessel.poly")]
    assert mesh.state(plan) == ("current", "")
    with open(work / "meshNoVessel.node", "a") as f:
        f.write("# edited\n")
    assert mesh.state(plan) == ("stale", "meshNoVessel.node changed since it was generated")


@pytest.mark.skipif(shutil.which("tetgen") is None, reason="tetgen not on PATH")
def test_status(tmp_path):
    work = tmp_path / "pro"
    _copy_ci_models(work)
    absent = pro("status", *TETGEN_MODEL, cwd=work)
    assert absent.returncode == 0, absent.stderr[-2000:]
    assert "mesh:      absent (no meshNoVessel.mesh.ymf)" in absent.stdout
    assert "pro run meshes on rank 0 first" in absent.stdout
    assert not (work / "meshNoVessel.mesh.ymf").exists()
    assert not (work / "meshNoVessel.ele").exists()

    assert pro("msh", *TETGEN_MODEL, "-l", "0", cwd=work).returncode == 0
    current = pro("status", *TETGEN_MODEL, cwd=work)
    assert "mesh:      current" in current.stdout
    assert "next:      pro run reads the mesh files; no meshing" in current.stdout

    stale = pro("status", *TETGEN_MODEL, "-C", "Refinement=1", cwd=work)
    assert "mesh:      stale (the generator's inputs or command changed)" in stale.stdout


def test_status_of_a_mesh_built_in_memory(tmp_path):
    shutil.copytree(os.path.join(HERE, "elliptic_redist", "RDLS"), tmp_path / "rdls")
    out = pro("status", "vortex2D_so.py", cwd=tmp_path / "rdls")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "domain:    RectangularDomain" in out.stdout
    assert "nothing to generate" in out.stdout


POISSON_SPEC = """\
model:
  name: "Poisson"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [u]
    equation_formulation: "Poisson"
    equations: ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    domain: "Ω = [0, 1] × [0, 1]"
    boundary_regions: [{name: wall, where: "∂Ω"}]
    boundary_conditions: [{region: wall, variable: u, type: dirichlet, value: 0.0}]
  weak_forms:
    - label: galerkin
      provenance: human_specified
      derivation: "integrate by parts"
      solution_spaces: {trial: H1, test: H1_0}
      bilinear: "∫ ∇u·∇v"
      linear: "∫ f v"
      stabilization_method: none
solution_paths:
  analytical:
    - name: exact
      provenance: human_specified
      method: closed_form
      solution: {formula: "u = sin(πx) sin(πy)"}
  discretizations:
    - name: P1
      provenance: human_specified
      from_weak_form: galerkin
      finite_element: {fields: {family: CG, order: 1}}
      solver: {type: linear}
"""


def test_spec_goes_to_the_ymf_runner(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    (tmp_path / "poisson.ymf").write_text(POISSON_SPEC, encoding="utf-8")
    out = pro("run", "poisson.ymf", "--cells", "4", "--emit-pn",
              "--outdir", "out", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    archive = (tmp_path / "out" / "poisson.archive.ymf").read_text()
    assert "set_by: pro run --cells 4" in archive
    emitted = sorted(p.name for p in (tmp_path / "out").glob("*_so.py"))
    assert len(emitted) == 1

    # the lean-tier route: the emitted files run with pro run, no spec needed
    work = tmp_path / "compute"
    work.mkdir()
    stem = emitted[0][:-len("_so.py")]
    for suffix in ("_p.py", "_n.py", "_so.py"):
        shutil.copy(tmp_path / "out" / (stem + suffix), work)
    ran = pro("run", stem + "_so.py", cwd=work)
    assert ran.returncode == 0, ran.stderr[-2000:]


def test_spec_without_the_front_end_explains_the_route(monkeypatch, capsys):
    import proteus.cli as cli
    monkeypatch.setattr(cli, "missing", lambda *names: ["sympy"])
    assert cli.main(["run", "spec.ymf", "--cells", "4"]) == 2
    err = capsys.readouterr().err
    assert "lacks sympy" in err
    assert "pro run SPEC.ymf --emit-pn" in err


LADR = ["ladr_2d_p.py", "ladr_2d_n.py"]


def _copy_ladr(directory):
    directory.mkdir()
    for f in ("adr.py",) + tuple(LADR):
        shutil.copy(os.path.join(HERE, "ci", f), directory)


@pytest.mark.skipif(shutil.which("parun") is None, reason="parun not on PATH")
def test_shell_batch_run_matches_parun(tmp_path):
    import numpy as np
    _copy_ladr(tmp_path / "shell")
    (tmp_path / "shell" / "batch.py").write_text(
        "solution = run()\n"
        "print('ran', type(solution).__name__, status, [p.name for p in pList])\n")
    out = pro("shell", *LADR, "--batch", "batch.py", cwd=tmp_path / "shell")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ran NS_base 0 ['ladr_2d']" in out.stdout

    _copy_ladr(tmp_path / "parun")
    ran = subprocess.run(["parun"] + LADR, cwd=tmp_path / "parun",
                         capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr[-2000:]
    expected = _datasets(tmp_path / "parun" / "ladr_2d.h5")
    actual = _datasets(tmp_path / "shell" / "ladr_2d.h5")
    assert sorted(actual) == sorted(expected)
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name


def test_shell_prompt_reads_python(tmp_path):
    _copy_ladr(tmp_path / "shell")
    out = subprocess.run(
        [sys.executable, "-m", "proteus.cli", "shell", *LADR, "-l", "0"],
        cwd=tmp_path / "shell", input="print('model is', so.name, opts.hotStart)\n",
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "model is ladr_2d False" in out.stdout


def test_shell_without_a_model(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "proteus.cli", "shell", "-l", "0"], cwd=tmp_path,
        input="run()\n", capture_output=True, text=True)
    assert "no model loaded: load('model_so.py') first" in out.stderr


@pytest.mark.skipif(shutil.which("triangle") is None, reason="triangle not on PATH")
def test_emitted_region_files_mesh_with_pro_msh(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    box = '    domain: "Ω = [0, 1] × [0, 1]"\n'
    assert box in POISSON_SPEC
    spec = POISSON_SPEC.replace(box, '''    domain:
      kind: piecewise_linear
      dimension: 2
      vertices: [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
      segments: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
''').replace("      solver: {type: linear}\n",
             "      solver: {type: linear}\n      mesh: {h: 0.25}\n")
    (tmp_path / "lshape.ymf").write_text(spec, encoding="utf-8")
    out = pro("run", "lshape.ymf", "--emit-pn", "--outdir", "out", cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    (so,) = (tmp_path / "out").glob("*_so.py")
    stem = so.name[:-len("_so.py")]

    # on the compute side: only the emitted files
    work = tmp_path / "compute"
    work.mkdir()
    for suffix in ("_p.py", "_n.py", "_so.py"):
        shutil.copy(tmp_path / "out" / (stem + suffix), work)
    status = pro("status", so.name, cwd=work)
    assert "generator: triangle" in status.stdout
    assert "mesh:      absent" in status.stdout
    made = pro("msh", so.name, "-l", "0", cwd=work)
    assert made.returncode == 0 and "running triangle -VApq30Dena" in made.stdout
    ran = pro("run", so.name, "--no-mesh", cwd=work)
    assert ran.returncode == 0, ran.stderr[-2000:]


@pytest.mark.parametrize("command", ["msh", "status"])
def test_msh_and_status_point_a_spec_at_its_emitted_files(tmp_path, command):
    (tmp_path / "spec.ymf").write_text("model: {}\n")
    out = pro(command, "spec.ymf", cwd=tmp_path)
    assert out.returncode == 2
    assert "pro run spec.ymf --emit-pn" in out.stderr

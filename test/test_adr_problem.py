"""proteus.ADRProblem: Proteus problems from a plain-data ADR form.

The unit tests build the ADR form by hand -- ADRProblem never needs sympy.
The end-to-end tests go through scripts/ymf_run and need ymf[symbolic].
"""
import os
import subprocess
import sys

import numpy
import pytest

from proteus import ADRProblem

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "ymf_run")


def poisson_adr(diffusion_depends_on=None):
    """-div(a grad u) = 1 on the unit square, written out as ymf emits it."""
    return {
        "dim": 2, "components": ["u"],
        "equations": [{
            "component": "u",
            "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1], "a": ["1.0", "1.0"],
                                "da": {}, "depends_on": diffusion_depends_on or {}}},
            "reaction": {"r": "-1.0", "dr": {}, "depends_on": {}},
        }],
    }


def test_constant_coefficients_get_proteus_constant_flags():
    c = ADRProblem.ADRCoefficients(poisson_adr())
    assert c.diffusion == {0: {0: {0: 'constant'}}}
    assert c.reaction == {0: {0: 'constant'}}
    assert c.potential == {0: {0: 'u'}}
    assert c.sdInfo[(0, 0)][0].tolist() == [0, 1, 2]


def test_any_dependence_makes_diffusion_nonlinear():
    # Proteus's diffusion flags are only 'constant' or 'nonlinear'
    c = ADRProblem.ADRCoefficients(poisson_adr({"u": "linear"}))
    assert c.diffusion == {0: {0: {0: 'nonlinear'}}}


def test_coefficients_are_evaluated_from_their_code_strings():
    adr = poisson_adr()
    adr["equations"][0]["reaction"]["r"] = "-x*y"
    c = ADRProblem.ADRCoefficients(adr)
    x = numpy.random.default_rng(0).random((3, 4, 3))
    q = {'x': x, ('u', 0): numpy.zeros((3, 4)), ('r', 0): numpy.zeros((3, 4)),
         ('a', 0, 0): numpy.zeros((3, 4, 2))}
    c.evaluate(0.0, q)
    numpy.testing.assert_array_equal(q[('r', 0)], -x[..., 0] * x[..., 1])
    assert (q[('a', 0, 0)] == 1.0).all()


def test_reorder_permutes_components_and_their_equations_together():
    problem = {"adr": {"dim": 2, "components": ["v_0", "v_1", "p"],
                       "equations": [{"component": n} for n in ("v_0", "v_1", "p")]},
               "unknowns": {"v": ["v_0", "v_1"], "p": ["p"]}}
    out = ADRProblem.reorder(problem, ["p", "v_0", "v_1"])
    assert out["adr"]["components"] == ["p", "v_0", "v_1"]
    assert [e["component"] for e in out["adr"]["equations"]] == ["p", "v_0", "v_1"]
    assert problem["adr"]["components"] == ["v_0", "v_1", "p"]      # untouched


def test_supg_pspg_refuses_systems_that_are_not_velocity_pressure():
    with pytest.raises(ValueError, match="velocity-pressure"):
        ADRProblem.velocity_pressure({"unknowns": {"u": ["u"], "w": ["w"]}})


# --- end to end, through ymf_run ---------------------------------------------------


@pytest.fixture
def poisson_spec(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    path = tmp_path / "poisson.ymf"
    path.write_text('''
Problem:
  name: "Poisson"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [u]
    equation_formulation: "Poisson"
    equations: ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    domain: "Ω = [0, 1] × [0, 1]"
    boundary_regions: [{name: wall, geometry: "∂Ω"}]
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
''', encoding="utf-8")
    return path


def ymf_run(*args):
    return subprocess.run([sys.executable, SCRIPT, *map(str, args)],
                          capture_output=True, text=True)


def rows(stdout):
    return [line.split() for line in stdout.splitlines() if line.strip()[:1].isdigit()]


def test_a_spec_runs_end_to_end_into_its_archive(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--cells", "4", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    table = rows(out.stdout)
    assert [r[0] for r in table] == ["4", "8"] and [r[-1] for r in table] == ["new", "new"]
    assert 1.8 < float(table[1][2]) < 2.2          # P1 in L2
    from ymf import closure
    spec, outputs, _ = closure.load(tmp_path / "poisson.archive.ymf")
    assert spec["Problem"]["strong_form"]["equations"] == ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    assert spec["solution_paths"]["discretizations"][0]["mesh"] == {"cells": [4, 8]}
    assert spec["composition"]["overrides"][0]["set_by"] == "ymf_run --cells 4 8"
    key = [k for k in outputs if "/cells=8/" in k][0]
    output = outputs[key]
    assert output["realization"] == {"cells": 8, "levels": 1}
    assert output["input"]["sha256"].startswith(key.rsplit("/", 1)[1])
    discrete = output["transformations"]["adr_to_discrete"]
    assert discrete["spaces"] == {"u": "C0_AffineLinearOnSimplexWithNodalBasis"}
    assert discrete["mesh"] == {"nodes_per_side": 9, "levels": 1}
    assert set(output["verification"]["l2_errors"]) == {"u"}
    files = sorted(p.name for p in tmp_path.iterdir() if p.suffix in (".h5", ".xmf"))
    assert files == sorted(closure.file_stem("poisson", k) + s
                           for k in outputs for s in (".h5", ".xmf"))


def test_like_make_it_computes_only_what_is_missing(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    out = ymf_run(poisson_spec, "--cells", "4", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert [r[-1] for r in rows(out.stdout)] == ["archive", "new"]    # "in the archive"
    assert "2 outputs, 1 new" in out.stdout


def test_an_archive_reruns_to_bitwise_identical_results(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    archive = tmp_path / "poisson.archive.ymf"
    out = ymf_run(archive, "--check")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "reproduced, bitwise" in out.stdout
    from ymf import closure
    _, outputs, _ = closure.load(archive)
    (output,) = outputs.values()
    assert len(output["reproduced"]) == 1


def test_a_differing_rerun_is_refused_unless_kept(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    archive = tmp_path / "poisson.archive.ymf"
    # tamper with the record, as a different machine's result would differ
    from ymf.archive import read_document, write_document
    document = read_document(archive)
    (output,) = document["outputs"].values()
    output["verification"]["l2_errors"]["u"] *= 1.5
    write_document(archive, document)
    out = ymf_run(archive, "--check")
    assert out.returncode == 1 and "NOT REPRODUCED" in out.stdout and "L2(u)" in out.stdout
    out = ymf_run(archive, "--check", "--keep-different")
    assert out.returncode == 0, out.stdout[-2000:]
    keys = list(read_document(archive)["outputs"])
    assert len(keys) == 2 and keys[1] == keys[0] + "~2"


def test_an_edited_spec_does_not_silently_replace_the_archive(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    poisson_spec.write_text(poisson_spec.read_text().replace('"Poisson"', '"Poisson, renamed"'))
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path)
    assert out.returncode != 0 and "earlier version of the input" in out.stdout
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path, "--prune")
    assert out.returncode == 0 and "1 outputs, 1 new" in out.stdout


def test_an_inline_multilevel_run_is_self_contained_and_reproducible(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--cells", "2", "--levels", "3", "--inline", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    from ymf import closure
    from ymf.archive import domain_arrays
    archive = tmp_path / "poisson.archive.ymf"
    _, outputs, _ = closure.load(archive)
    ((key, output),) = outputs.items()
    assert "/cells=2/levels=3/" in key and output["storage"] == "inline"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["poisson.ymf", "poisson.archive.ymf", closure.file_stem("poisson", key) + ".xmf"])
    nodes = [a for k, a in domain_arrays(output["approximation"]).items()
             if k.endswith("Geometry")][-1]
    assert nodes.shape == (81, 3)        # the finest level: 2 cells refined twice -> 8x8
    out = ymf_run(archive, "--check")
    assert out.returncode == 0 and "reproduced, bitwise" in out.stdout, out.stdout[-2000:]


def test_emitted_pn_files_reproduce_the_output_bitwise_under_parun(poisson_spec, tmp_path):
    import numpy
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path, "--emit-pn")
    assert out.returncode == 0, out.stderr[-2000:]
    from ymf import closure
    from ymf.archive import domain_arrays, read_ymf
    _, outputs, _ = closure.load(tmp_path / "poisson.archive.ymf")
    ((key, output),) = outputs.items()
    stem = closure.file_stem("poisson", key)
    work = tmp_path / "pn"
    work.mkdir()
    for suffix in ("_p", "_n", "_so"):
        (work / (stem + suffix + ".py")).write_text((tmp_path / (stem + suffix + ".py")).read_text())
    parun = os.path.join(os.path.dirname(sys.executable), "parun")
    run = subprocess.run([parun, stem + "_so.py"], cwd=work, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    mine = domain_arrays(read_ymf(work / (stem + "_pn.ymf"))[0], work)
    recorded = domain_arrays(output["approximation"], tmp_path)
    assert set(mine) == set(recorded)
    assert all(numpy.array_equal(mine[k], recorded[k]) for k in recorded)


@pytest.fixture
def kovasznay_spec(tmp_path):
    """Kovasznay flow, Taylor-Hood, velocity given all round: p up to a constant."""
    pytest.importorskip("ymf.symbolic")
    path = tmp_path / "kovasznay.ymf"
    path.write_text('''
Problem:
  name: "Kovasznay"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [{name: v, rank: 1}, p]
    equation_formulation: "Navier-Stokes"
    equations:
      - "∇·(ρ v⊗v) − μΔv + ∇p = 0  in Ω"
      - "∇·v = 0  in Ω"
    domain: "Ω = [-0.5, 1.0] × [-0.5, 1.5]"
    boundary_regions: [{name: outer, geometry: "∂Ω"}]
    boundary_conditions:
      - {region: outer, variable: v, type: dirichlet,
         formula: "(1 - exp(λx) cos(2πy), λ/(2π) exp(λx) sin(2πy))"}
    coefficients:
      ρ: {value: 1.0}
      μ: {value: 0.025}
      λ: "ρ/(2μ) - sqrt(ρ²/(4μ²) + 4π²)"
  weak_forms:
    - label: mixed
      provenance: human_specified
      derivation: "integrate by parts"
      solution_spaces: {trial: "H1² × L2", test: "H1_0² × L2"}
      bilinear: "..."
      linear: "..."
      stabilization_method: none
solution_paths:
  analytical:
    - name: exact
      provenance: human_specified
      method: closed_form
      solution:
        formula: |-
          v = (1 - exp(λx) cos(2πy), λ/(2π) exp(λx) sin(2πy))
          p = (1 - exp(2λx))/2
  discretizations:
    - name: taylor_hood
      provenance: human_specified
      from_weak_form: mixed
      finite_element: {velocity: {family: CG, order: 2}, pressure: {family: CG, order: 1}}
      mesh: {cells: [4, 8]}
      solver: {type: nonlinear}
''', encoding="utf-8")
    return path


def test_a_pressure_up_to_a_constant_is_solved_in_the_null_space(kovasznay_spec, tmp_path):
    import numpy
    out = ymf_run(kovasznay_spec, "--outdir", tmp_path, "--emit-pn")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "L2(p)*" in out.stdout
    table = rows(out.stdout)
    assert 2.5 < float(table[1][2]) and 1.5 < float(table[1][6])      # v_0 P2, p P1 rates
    from ymf import closure
    from ymf.archive import domain_arrays, read_ymf
    archive = tmp_path / "kovasznay.archive.ymf"
    _, outputs, _ = closure.load(archive)
    output = outputs[[k for k in outputs if "/cells=4/" in k][0]]
    assert output["verification"]["modulo_constants"] == ["p"]
    discrete = output["transformations"]["adr_to_discrete"]
    assert discrete["null_space"] == {"constant": ["p"]}
    assert discrete["linear_solver"] == "KSP_petsc4py"
    assert output["transformations"]["solve"]["by"]["threads"]["OMP_NUM_THREADS"] == "1"
    check = ymf_run(archive, "--check")
    assert check.returncode == 0 and check.stdout.count("reproduced, bitwise") == 2, check.stdout[-2000:]
    # and the emitted files reproduce it under parun, single-threaded
    stem = closure.file_stem("kovasznay", [k for k in outputs if "/cells=4/" in k][0])
    work = tmp_path / "pn"
    work.mkdir()
    for suffix in ("_p", "_n", "_so"):
        (work / (stem + suffix + ".py")).write_text((tmp_path / (stem + suffix + ".py")).read_text())
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    parun = os.path.join(os.path.dirname(sys.executable), "parun")
    run = subprocess.run([parun, stem + "_so.py"], cwd=work, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    mine = domain_arrays(read_ymf(work / (stem + "_pn.ymf"))[0], work)
    recorded = domain_arrays(output["approximation"], tmp_path)
    assert all(numpy.array_equal(mine[k], recorded[k]) for k in recorded)

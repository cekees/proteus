"""``pro spec``: check a YMF spec, and say how far it has got.

The linter for the first step of ``pro spec`` -> ``pro weak`` -> ``pro dis``
-> ``pro run``. It writes nothing. It loads the document as the runner
would, then reports, each with where it was found:

* **schema** and **composition**: the loader's own errors;
* **state**: the ``ymf_state`` the content reaches, why it is not the next
  one, and a declared state the content does not reach;
* **units**: ``ymf.units.check``, when pint is installed;
* **adr**: whether the strong form becomes a Proteus ADR problem -- the
  notation parses, the equations classify -- and what that problem is,
  including an unknown it determines only up to a constant;
* **discretization**: what ``pro run`` would refuse or skip in each branch
  (element spaces, stabilization, linear solver);
* **generated**: for a file a tool wrote, whether its input changed since
  (stale) or its content was edited (see ``ymf.generated``).

It exits 1 when there is an error, or when the document is short of the
state ``--require`` asks for (well-posed unless told otherwise), so a
script can gate on it. ``--json`` gives the same report as data, for an
agent driving the steps.
"""
import json
import sys

from . import json_stdout, missing, parser_kwargs

__all__ = ["add_parser", "run", "lint", "NEEDS"]

#: what checking a spec imports (ymf[symbolic], which brings strictyaml)
NEEDS = ("ymf.symbolic", "sympy", "strictyaml")

#: the states in order, so a requirement can be compared (ymf.state.STATES)
_ORDER = ("abstract", "well-posed", "discretized", "closed")


def add_parser(subparsers):
    p = subparsers.add_parser(
        "spec",
        help="check a YMF spec and say how far it has got (writes nothing)",
        description="Check a YMF spec: schema, composition, units, notation, "
                    "whether it becomes a Proteus problem, and what pro run "
                    "would refuse in each discretization. Reports the "
                    "ymf_state it reaches. Writes nothing.",
        **parser_kwargs())
    p.add_argument("spec", metavar="SPEC.ymf", help="a YMF spec or archive")
    p.add_argument("--json", action="store_true",
                   help="print the report as JSON")
    p.add_argument("--require", choices=_ORDER, default="well-posed",
                   help="exit 1 below this state (default: well-posed)")
    p.add_argument("--missing-units", choices=("skip", "warn", "error"),
                   default="skip",
                   help="what a boundary condition whose units cannot be "
                        "checked for want of declared ones becomes")
    p.set_defaults(handler=run)
    return p


def _issue(severity, check, location, message):
    return {"severity": severity, "check": check, "location": location,
            "message": message}


def _schema_location(exc, path):
    """``file:line`` of a strictyaml error, where it has one."""
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    if mark is not None and getattr(mark, "line", None) is not None:
        return "%s:%d" % (getattr(mark, "name", None) or path, mark.line + 1)
    return str(path)


def _first_line(text):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return " ".join(lines[-2:]) if lines else str(text)


def lint(path, missing_units="skip"):
    """The report for one spec, as plain data (see the module docstring)."""
    import strictyaml
    from ymf.compose import YmfCompositionError, load_composed
    from ymf.state import reached

    report = {"file": str(path), "name": None, "declared": None, "state": None,
              "next": None, "issues": [], "adr": None, "discretizations": {}}
    issues = report["issues"]
    try:
        doc = load_composed(path, check_state=False)
    except strictyaml.YAMLValidationError as exc:
        issues.append(_issue("error", "schema", _schema_location(exc, path),
                             _first_line(exc)))
        return report
    except YmfCompositionError as exc:
        issues.append(_issue("error", "composition", str(path), str(exc)))
        return report
    report["name"] = (doc.get("model") or {}).get("name")
    declared = doc.get("ymf_state")
    report["declared"] = declared
    well_posed = declared != "abstract"
    state, why = reached(doc, well_posed)
    report["state"], report["next"] = state, (why or None)
    if declared and _ORDER.index(declared) > _ORDER.index(state):
        issues.append(_issue("error", "state", str(path),
                             "declares ymf_state: %s, but it is only %s: %s"
                             % (declared, state, why)))
    _check_generated(path, issues)
    if not well_posed:
        return report

    if missing("pint"):
        issues.append(_issue("info", "units", str(path),
                             "not checked: pint is not installed (pip install 'ymf[units]')"))
    else:
        from ymf.units import check as check_units
        for found in check_units(doc, missing_units=missing_units):
            issues.append(_issue(found.severity, "units", found.location, found.message))

    from ymf.symbolic import adr_problem
    try:
        problem = adr_problem(doc)
    except (ValueError, KeyError) as exc:
        issues.append(_issue("error", "adr", "model.strong_form",
                             str(exc) if isinstance(exc, ValueError)
                             else "%s: %s" % (type(exc).__name__, exc)))
        return report
    report["adr"] = {
        "unknowns": {name: list(comps) for name, comps in problem["unknowns"].items()},
        "dimension": problem["adr"]["dim"],
        "time_dependent": bool(problem.get("time")),
        "up_to_constant": list(problem.get("up_to_constant") or []),
        "periodic": sorted({e["component"] for e in problem.get("periodic") or []}),
    }
    for name in problem.get("up_to_constant") or []:
        issues.append(_issue("info", "adr", "model.strong_form",
                             "%s is determined only up to a constant: solved in the "
                             "null space, errors reported modulo constants" % (name,)))

    for entry in (doc.get("solution_paths") or {}).get("discretizations") or []:
        name = entry.get("name")
        report["discretizations"][name] = _check_discretization(
            doc, problem, entry, issues)
    return report


def _check_discretization(doc, problem, entry, issues):
    """What pro run would do with one branch: ``runs``, or why it would not."""
    from proteus.ADRProblem import STABILIZATIONS, linear_solver_choice
    from ymf import closure
    from .ymfrun import element_spaces, stabilization_of
    where = "solution_paths.discretizations[%s]" % (entry.get("name"),)
    try:
        spaces = element_spaces(problem, entry.get("finite_element") or {})
    except SystemExit as exc:            # the runner's own refusal
        issues.append(_issue("error", "discretization", where, str(exc)))
        return "refused"
    method = stabilization_of(doc, entry)
    if method not in STABILIZATIONS:
        issues.append(_issue("warning", "discretization", where,
                             "pro run skips it: its weak form needs %s stabilization, "
                             "which is not wired up (have %s)"
                             % (method, ", ".join(STABILIZATIONS))))
        return "skipped"
    solver = entry.get("solver") or {}
    try:
        linear_solver_choice(problem, {c: tuple(s) for c, s in spaces.items()},
                             solver.get("linear_solver"), solver.get("petsc_options"))
    except ValueError as exc:
        issues.append(_issue("warning", "discretization", where,
                             "pro run skips it: %s" % (exc,)))
        return "skipped"
    try:
        planned = closure.realizations(entry)
    except ValueError as exc:
        issues.append(_issue("error", "discretization", where, str(exc)))
        return "refused"
    if not planned:
        return "runs, given a mesh (--cells or --h)"
    return "runs: %d output%s" % (len(planned), "" if len(planned) == 1 else "s")


def _check_generated(path, issues):
    """Stale or edited, for a file a tool wrote; nothing for any other."""
    import importlib.util
    if importlib.util.find_spec("ymf.generated") is None:
        return
    from ymf import generated
    try:
        status = generated.check(path)
    except ValueError as exc:
        issues.append(_issue("error", "generated", str(path), str(exc)))
        return
    if status.record is None:
        return
    for reason in status.reasons:
        issues.append(_issue("warning", "generated", str(path), reason))


def _next_step(report):
    """What stands between the document and the next state, briefly.

    The JSON report keeps ``next`` whole; here a list of every missing
    output key would bury the rest, so it is counted instead.
    """
    import re
    state, why, declared = report["state"], report["next"], report["declared"]
    if declared == "abstract":
        return " (declared): a model to be extended into a well-posed spec"
    if declared and declared != state:
        prefix = " (declares %s)" % (declared,)
    else:
        prefix = ""
    if not why:
        return prefix
    missing = re.match(r"(\d+) of (\d+) planned outputs (?:is|are) missing", why)
    if missing:
        n, total = int(missing.group(1)), int(missing.group(2))
        return prefix + " -- %d of %d planned outputs to compute: pro run %s" % (
            n, total, report["file"])
    return prefix + " -- not further: " + why


def _render(report):
    lines = ["%s%s" % (report["file"], ": " + report["name"] if report["name"] else "")]
    if report["state"] is not None:
        lines.append("state: " + report["state"] + _next_step(report))
    adr = report["adr"]
    if adr:
        unknowns = ", ".join("%s (%s)" % (n, ", ".join(c)) for n, c in adr["unknowns"].items())
        lines.append("problem: %d-D, %s, unknowns %s"
                     % (adr["dimension"], "time-dependent" if adr["time_dependent"]
                        else "steady", unknowns))
    for name, outcome in report["discretizations"].items():
        lines.append("  %s: %s" % (name, outcome))
    for i in report["issues"]:
        lines.append("[%s] %s: %s: %s" % (i["severity"], i["check"], i["location"],
                                          i["message"]))
    return "\n".join(lines)


def _exit_status(report, require):
    if any(i["severity"] == "error" for i in report["issues"]):
        return 1
    if report["state"] is None or _ORDER.index(report["state"]) < _ORDER.index(require):
        return 1
    return 0


def run(args):
    absent = missing(*NEEDS)
    if absent:
        print("pro spec: needs {0}, which this environment lacks.\n"
              "Install the front end with: pip install 'ymf[symbolic]'"
              .format(", ".join(absent)), file=sys.stderr)
        return 2
    # Writes nothing: not even the proteus_default.log Profiling opens at
    # exit when no log was ever opened. Setting procID is what
    # runtime.start does with log=None, without starting MPI and PETSc.
    # before anything imports mpi4py: see json_stdout
    out = json_stdout() if args.json else sys.stdout
    from proteus import Profiling
    if Profiling.procID is None:
        Profiling.procID = 0
    report = lint(args.spec, missing_units=args.missing_units)
    status = _exit_status(report, args.require)
    if args.json:
        print(json.dumps(dict(report, exit_status=status), indent=2, ensure_ascii=False),
              file=out, flush=True)
    else:
        print(_render(report))
    return status

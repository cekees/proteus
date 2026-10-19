"""``pro dis``: the discretizations a spec's weak forms are solved with, written out.

The third step of ``pro spec`` -> ``pro weak`` -> ``pro dis`` -> ``pro
run``. ``pro dis F`` reads a well-posed spec that has its weak forms
(usually ``<stem>.weak.ymf``, from ``pro weak``) and writes
``<stem>.disc.ymf``: a file that ``extends`` it, records that it was
generated (``ymf.generated``), and declares ``ymf_state: discretized``,
which the loader checks. ``pro run <stem>.disc.ymf`` then runs it.

What it writes:

* **A discretization for each weak form that has none,** among those
  Proteus can solve (``ADRProblem.check_stabilization``), with the
  defaults of :mod:`proteus.cli.defaults`: the elements for the
  formulation, a mesh, the solver, and backward Euler for a problem with a
  time interval. It names its weak form (``from_weak_form``) even where
  that could be left out, so it says which formulation it solves.
* **What an existing discretization lacks to be discretized:** a mesh, and
  the elements when it gives none. Nothing it does give is changed.

``--cells``, ``--h`` and ``--levels`` replace the default mesh (a list is a
study), ``--only`` limits the new discretizations to some weak forms, and
``--list`` says what would be written and writes nothing. An existing
``<stem>.disc.ymf`` is regenerated only if a tool wrote it and nobody has
edited it since, as with ``pro weak``.

A spec with no weak form is refused, with the command that writes them:
which formulation a discretization solves is a choice ``pro dis`` does not
make silently.
"""
import json
import os
import sys

from . import json_stdout, missing, parser_kwargs

__all__ = ["add_parser", "run", "plan", "write", "NEEDS"]

NEEDS = ("ymf.symbolic", "ymf.generated", "sympy", "strictyaml")


def add_parser(subparsers):
    p = subparsers.add_parser(
        "dis",
        help="write the discretizations a spec's weak forms are solved with",
        description="Write a discretization for each weak form of a YMF spec "
                    "that has none, and a mesh for each one that lacks it, to "
                    "<stem>.disc.ymf, a file that extends the spec and is "
                    "ymf_state: discretized. --list says what would be "
                    "written, and writes nothing.",
        **parser_kwargs())
    p.add_argument("spec", metavar="SPEC.ymf",
                   help="a well-posed YMF spec with its weak forms (pro weak writes them)")
    p.add_argument("--list", action="store_true",
                   help="say what would be written, and write nothing")
    p.add_argument("--only", nargs="+", metavar="LABEL",
                   help="write discretizations for these weak forms only")
    p.add_argument("--cells", type=int, nargs="+", metavar="N",
                   help="cells per side of a box mesh (several: a study)")
    p.add_argument("--h", type=float, nargs="+", metavar="H",
                   help="element size of a region's mesh (several: a study)")
    p.add_argument("--levels", type=int, nargs="+", metavar="L",
                   help="levels of uniform refinement (several: a study)")
    p.add_argument("-o", "--output", metavar="FILE",
                   help="where to write (default: <stem>.disc.ymf beside the spec)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a file that was edited since, or not generated")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(handler=run)
    return p


# --- what to write ----------------------------------------------------------------


def _one_or_list(values):
    return None if not values else (values[0] if len(values) == 1 else list(values))


def _mesh(problem, cells=None, h=None, levels=None):
    """The mesh block: the command line's, else the default (proteus.cli.defaults)."""
    from . import defaults
    box = problem["domain"].get("region") is None
    if cells and not box:
        raise ValueError("the domain is a region, meshed by its element size: give --h, "
                         "not --cells")
    if h and box:
        raise ValueError("the domain is a box, meshed by cells per side: give --cells, "
                         "not --h")
    block = ({"cells": _one_or_list(cells)} if cells else
             {"h": _one_or_list(h)} if h else defaults.mesh(problem))
    if levels:
        block["levels"] = _one_or_list(levels)
    return block


def _name(label, short):
    """A discretization's name: its weak form's label and its elements."""
    base = label[len("proteus_"):] if label.startswith("proteus_") else label
    return "%s_%s" % (base, short)


def _supported(problem, method):
    """None if Proteus solves this weak form's method, else why not."""
    from proteus.ADRProblem import check_stabilization
    try:
        check_stabilization(problem, (method or "none"))
        return None
    except ValueError as exc:
        return str(exc)


def _new_entry(problem, weak, mesh_block):
    from . import defaults
    method = (weak.get("stabilization_method") or "none").strip()
    finite_element, short = defaults.elements(problem, method)
    solver, dofs, note = defaults.solver(problem, finite_element, mesh_block)
    entry = {
        "name": _name(weak["label"], short),
        "provenance": "proteus_derived",
        "from_weak_form": weak["label"],
        "finite_element": finite_element,
        "mesh": mesh_block,
        "solver": solver,
    }
    time = defaults.time(problem)
    if time is not None:
        entry["discretization"] = time
    details = ["Proteus's defaults for %s (proteus.cli.defaults): about %d unknowns on "
               "the finest mesh." % (method if method.lower() != "none" else "Galerkin",
                                     dofs)]
    if note:
        details.append(note + ".")
    entry["discretization_details"] = " ".join(details)
    return entry


def plan(spec, only=None, cells=None, h=None, levels=None):
    """``(report, parent, entries)`` for a spec; writes nothing.

    ``entries`` are what the file would hold: new discretizations, whole,
    and for an existing one only the fields it lacks (with its name, so
    composition merges them into it).
    """
    from ymf.compose import load_composed
    from ymf.symbolic import adr_problem
    parent = load_composed(spec, check_state=False)
    parent.pop("outputs", None)
    report = {"spec": str(spec), "name": (parent.get("model") or {}).get("name"),
              "discretizations": [], "skipped": [], "written": None, "generated": None,
              "error": None}
    if parent.get("ymf_state") == "abstract":
        report["error"] = "%s is abstract: extend it into a well-posed spec first" % (spec,)
        return report, None, []
    weak_forms = (parent.get("model") or {}).get("weak_forms") or []
    if not weak_forms:
        report["error"] = ("%s has no weak form, so nothing says which formulation a "
                           "discretization solves; run: pro weak %s" % (spec, spec))
        return report, parent, []
    labels = [w["label"] for w in weak_forms]
    unknown = sorted(set(only or ()) - set(labels))
    if unknown:
        report["error"] = "no weak form %s (have %s)" % (", ".join(unknown), ", ".join(labels))
        return report, parent, []
    problem = adr_problem(parent)
    try:
        mesh_block = _mesh(problem, cells, h, levels)
    except ValueError as exc:             # the command line's, not the spec's
        report["error"] = str(exc)
        return report, parent, []
    existing = (parent.get("solution_paths") or {}).get("discretizations") or []
    used = {d.get("from_weak_form") for d in existing}
    entries = []
    for d in existing:                     # what an existing one lacks
        fill = {}
        if not d.get("mesh"):
            fill["mesh"] = mesh_block
        if not d.get("finite_element"):
            from . import defaults
            weak = next((w for w in weak_forms if w["label"] == d.get("from_weak_form")), {})
            fill["finite_element"], _ = defaults.elements(
                problem, weak.get("stabilization_method") or "none")
        if fill:
            entries.append(dict({"name": d["name"]}, **fill))
            report["discretizations"].append({"name": d["name"], "new": False,
                                              "from_weak_form": d.get("from_weak_form"),
                                              "adds": sorted(fill)})
    names = {d["name"] for d in existing}
    for weak in weak_forms:                # a new one for each weak form with none
        label = weak["label"]
        if label in used or (only and label not in only):
            continue
        why = _supported(problem, weak.get("stabilization_method"))
        if why is not None:
            report["skipped"].append({"weak_form": label, "why": why})
            continue
        entry = _new_entry(problem, weak, mesh_block)
        if entry["name"] in names:
            report["error"] = ("there is already a discretization named %s; give its "
                               "weak form %s a mesh there, or rename it"
                               % (entry["name"], label))
            return report, parent, []
        names.add(entry["name"])
        entries.append(entry)
        report["discretizations"].append({
            "name": entry["name"], "new": True, "from_weak_form": label,
            "finite_element": entry["finite_element"], "mesh": entry["mesh"],
            "solver": entry["solver"], "discretization": entry.get("discretization")})
    if not entries:
        report["error"] = ("nothing to write: every weak form Proteus solves already "
                           "has a discretization with a mesh")
    return report, parent, entries


def _out_path(spec, output):
    from ymf.closure import archive_stem
    if output:
        return os.path.abspath(output)
    stem = archive_stem(spec)
    if stem.endswith(".weak"):
        stem = stem[:-len(".weak")]
    return os.path.join(os.path.dirname(os.path.abspath(spec)), stem + ".disc.ymf")


def write(spec, out, parent, entries, force=False):
    """Write ``out``; return its generated record. Raises ValueError to refuse."""
    import proteus
    import ymf
    from ymf import generated
    from ymf.compose import load_composed
    if os.path.exists(out) and not force:
        status = generated.check(out)
        if status.record is None:
            raise ValueError("%s exists and was not written by a tool; give -o, "
                             "or --force to overwrite it" % (out,))
        if status.edited:
            raise ValueError("%s was edited since %s wrote it; it is the editor's "
                             "now. Give -o, or --force to overwrite it"
                             % (out, status.record["by"]))
    extends = os.path.relpath(os.path.abspath(spec), os.path.dirname(out))
    record = generated.write(out, extends,
                             "pro dis (proteus %s, ymf %s)" % (proteus.__version__,
                                                                ymf.__version__),
                             "model_and_analytical", parent,
                             {"solution_paths": {"discretizations": entries}},
                             state="discretized")
    try:
        load_composed(out)                 # the full schema, and the declared state
    except Exception as exc:
        os.remove(out)
        raise ValueError("what pro dis would write does not load (%s); nothing was "
                         "written" % (exc,))
    return record


def _render(report):
    lines = ["%s%s" % (report["spec"], ": " + report["name"] if report["name"] else "")]
    for d in report["discretizations"]:
        if d["new"]:
            fe = d["finite_element"]
            spaces = ", ".join("%s %s%d" % (k, v["family"], v["order"]) for k, v in fe.items())
            mesh = ", ".join("%s %s" % kv for kv in d["mesh"].items())
            lines.append("  %-24s new: %s; %s; %s%s" % (
                d["name"], spaces, mesh, d["solver"].get("linear_solver"),
                "; backward Euler dt %g" % d["discretization"]["dt"]
                if d.get("discretization") else ""))
        else:
            lines.append("  %-24s adds %s" % (d["name"], ", ".join(d["adds"])))
    for s in report["skipped"]:
        lines.append("  %-24s skipped: %s" % (s["weak_form"], s["why"]))
    if report["written"]:
        lines.append("wrote %s" % (report["written"],))
    if report["error"]:
        lines.append("pro dis: " + report["error"])
    return "\n".join(lines)


def run(args):
    absent = missing(*NEEDS)
    if absent:
        print("pro dis: needs {0}, which this environment lacks.\n"
              "Install the front end with: pip install 'ymf[symbolic]'"
              .format(", ".join(absent)), file=sys.stderr)
        return 2
    stdout = json_stdout() if args.json else sys.stdout   # before MPI: see json_stdout
    from proteus import Profiling
    if Profiling.procID is None:          # no proteus_default.log (see pro spec)
        Profiling.procID = 0
    entries, parent = [], None
    try:
        report, parent, entries = plan(args.spec, args.only, args.cells, args.h, args.levels)
    except ValueError as exc:             # the loader's, the classifier's, the mesh's
        report = {"spec": args.spec, "name": None, "discretizations": [], "skipped": [],
                  "written": None, "generated": None,
                  "error": "%s (run pro spec %s)" % (exc, args.spec)}
    if not report["error"] and not args.list:
        out = _out_path(args.spec, args.output)
        try:
            report["generated"] = write(args.spec, out, parent, entries, force=args.force)
            report["written"] = out
        except ValueError as exc:
            report["error"] = str(exc)
    status = 1 if report["error"] else 0
    if args.json:
        print(json.dumps(dict(report, exit_status=status), indent=2, ensure_ascii=False),
              file=stdout, flush=True)
    else:
        print(_render(report))
    return status

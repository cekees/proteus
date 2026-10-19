"""``pro status``: where a model or a spec stands, and what to do next.

For a **model** (Python modules): what ``pro run`` would do with its mesh,
and why. It loads the model (with the same context options the run would
get), works out the mesh plan, and compares it with the manifest ``pro
msh`` or ``pro run`` wrote. It runs no mesh generator and writes no
manifest. Loading a model does run its modules, and those may write files
of their own, such as the domain's .poly.

For a **spec** (``SPEC.ymf``): the chain of files it is built from and the
files beside it that extend it (``pro weak``'s ``.weak.ymf``, ``pro
disc``'s ``.disc.ymf``), and for each: the ``ymf_state`` it reaches,
whether a tool wrote it and if so whether it is current, stale (its input
changed since) or hand-edited, its weak forms and discretizations, and how
many of its planned outputs its archive holds. Then the next command:
regenerate a stale file, ``pro weak``, ``pro disc`` or ``pro run``, or
nothing to do. It writes nothing; ``--json`` gives the report as data.
"""
import json
import os
import sys

from . import json_stdout, missing, parser_kwargs

__all__ = ["add_parser", "run", "describe", "spec_status"]

#: what reporting a spec's chain imports (ymf[spec]: strictyaml)
SPEC_NEEDS = ("ymf.generated", "ymf.compose", "strictyaml")


def add_parser(subparsers):
    p = subparsers.add_parser(
        "status",
        help="show whether the mesh is up to date and what pro run would do",
        description="Show a model's mesh plan, whether its mesh files are "
                    "up to date, and what pro run would do. Runs no mesh "
                    "generator (the model's own modules still run).",
        **parser_kwargs())
    p.add_argument("inputs", nargs="+", metavar="MODEL.py|SPEC.ymf",
                   help="so module, TwoPhaseFlow case module, or p and n modules; "
                        "or a YMF spec, for its chain of files")
    p.add_argument("--json", action="store_true",
                   help="for a spec: print the report as JSON")
    p.add_argument("-C", "--context", metavar="'NAME=VALUE ...'",
                   help="context options, as for pro run")
    p.add_argument("-o", "--context-file", metavar="FILE",
                   help="file of context options; -C entries override it")
    p.add_argument("-H", "--hot-start", action="store_true",
                   help="as pro run -H: the run would reuse the existing mesh")
    p.set_defaults(handler=run, pumi=False, petsc=None, petsc_file=None,
                   log_level=0, verbose=False)
    return p


def describe(args, domain, meshOptions, plan):
    """``(lines, next_step)``: the report, and what pro run would do."""
    lines = ["model:     " + " ".join(args.inputs),
             "domain:    " + type(domain).__name__]
    if args.hot_start:
        return lines, "pro run -H reads the existing mesh (hot start)"
    if plan is None:
        return lines, ("pro run builds the mesh in memory or reads the "
                       "mesh files the domain names; nothing to generate")
    from . import mesh
    lines += ["generator: " + plan["generator"],
              "command:   " + plan["command"],
              "inputs:    " + ", ".join(os.path.basename(p) for p in plan["inputs"]),
              "outputs:   " + ", ".join(os.path.basename(p) for p in plan["outputs"]),
              "manifest:  " + os.path.basename(mesh.manifest_path(plan))]
    if not meshOptions.genMesh:
        return lines, ("genMesh=False: pro run reads the mesh files as they "
                       "are and never regenerates them")
    state, reason = mesh.state(plan)
    lines.append("mesh:      " + state + (" (" + reason + ")" if reason else ""))
    if state == "current":
        return lines, "pro run reads the mesh files; no meshing"
    return lines, ("pro run meshes on rank 0 first; pro run --no-mesh "
                   "stops with an error; pro msh meshes without running")


# --- a spec: its chain of files ----------------------------------------------------


def _summary(path):
    """``{state, why, name, weak_forms, discretizations, doc}`` of a file, composed."""
    import collections
    from ymf.compose import load_composed
    from ymf.state import reached
    try:
        doc = load_composed(path, check_state=False)
    except Exception as exc:              # the schema's or composition's
        return {"state": None, "why": str(exc).strip().splitlines()[-1], "name": None,
                "weak_forms": {}, "discretizations": 0, "doc": None}
    doc.pop("outputs", None)
    state, why = reached(doc, doc.get("ymf_state") != "abstract")
    model = doc.get("model") or {}
    provenance = collections.Counter(w.get("provenance") for w in model.get("weak_forms") or [])
    return {"state": state, "why": why or None, "name": model.get("name"),
            "weak_forms": dict(provenance),
            "discretizations": len((doc.get("solution_paths") or {}).get("discretizations") or []),
            "doc": doc}


def _archive(path, doc):
    """``{file, present, planned}`` for the archive pro run writes for ``path``, or None."""
    from ymf import closure
    stem = closure.archive_stem(path)
    for where in (os.path.dirname(os.path.abspath(path)), os.getcwd()):
        archive = os.path.join(where, stem + ".archive.ymf")
        if os.path.exists(archive):
            break
    else:
        return None
    planned = [key for key, _, _ in closure.planned_outputs(doc)]
    _, outputs, _ = closure.load(archive, check_state=False)
    kept, _ = closure.partition(doc, outputs)
    present = {closure._base_key(k) for k in kept}
    return {"file": archive, "planned": len(planned),
            "present": len([k for k in planned if k in present])}


def _next(leaf, chain_infos):
    """The next command for one leaf of the tree."""
    rel = os.path.relpath(leaf["file"])
    for info in chain_infos:                       # upstream first
        g = info["generated"]
        if g and "error" in g:
            return "fix %s: %s" % (os.path.relpath(info["file"]), g["error"])
        if g and g["stale"]:
            command = g["regenerate"] or "regenerate it"
            return "%s is stale; regenerate it: %s" % (os.path.relpath(info["file"]), command)
    if leaf["state"] is None:
        return "fix it: %s (pro spec %s says more)" % (leaf["why"], rel)
    if leaf["state"] == "abstract":
        return "extend it into a well-posed spec"
    if leaf["state"] == "well-posed":
        if not leaf["weak_forms"]:
            return "pro weak " + rel
        return "pro disc " + rel
    archive = leaf.get("archive")
    if archive and archive["planned"] and archive["present"] == archive["planned"]:
        return "nothing to do: every planned output is in %s" % (
            os.path.relpath(archive["file"]),)
    return "pro run " + rel


def spec_status(path):
    """The report for a spec: its chain, the files under it, and what to do next."""
    from . import chain
    infos = chain.chain(path)
    tree = chain.descendants(path)
    files = [i["file"] for i in infos] + [f for f in tree if f != os.path.abspath(path)]
    entries = {}
    for info in infos:
        entries[info["file"]] = info
    for f in files:
        if f not in entries:
            entries[f] = chain.file_info(f)
    for f, entry in entries.items():
        summary = _summary(f)
        doc = summary.pop("doc")
        entry.update(summary)
        if doc is not None and summary["state"] in ("discretized", "closed"):
            entry["archive"] = _archive(f, doc)
            if entry["archive"] and entry["archive"]["planned"] and \
                    entry["archive"]["present"] == entry["archive"]["planned"]:
                entry["state"] = "closed"
    leaves = [f for f in tree if not tree[f]]
    report = {"file": os.path.abspath(path), "name": entries[os.path.abspath(path)]["name"],
              "chain": [entries[i["file"]] for i in infos],
              "below": {f: children for f, children in tree.items() if children},
              "files": {f: entries[f] for f in files},
              "next": {}}
    for leaf in leaves:
        report["next"][leaf] = _next(entries[leaf],
                                     [entries[i["file"]] for i in chain.chain(leaf)])
    return report


def _describe_file(entry):
    g = entry["generated"]
    if g is None:
        written = "by hand"
    elif "error" in g:
        written = "generated record unreadable"
    else:
        written = "%s, %s" % (g["by"].split(" (")[0], "stale" if g["stale"] else
                              "hand-edited" if g["edited"] else "current")
    parts = [written]
    if entry.get("weak_forms"):
        n = sum(entry["weak_forms"].values())
        parts.append("weak forms: %d (%s)" % (n, ", ".join(
            "%s%s" % (p, "" if c == 1 else " x%d" % c) for p, c in entry["weak_forms"].items())))
    if entry.get("discretizations"):
        parts.append("%d discretization%s" % (entry["discretizations"],
                                              "" if entry["discretizations"] == 1 else "s"))
    archive = entry.get("archive")
    if archive:
        parts.append("archive %s: %d of %d outputs" % (
            os.path.basename(archive["file"]), archive["present"], archive["planned"]))
    return "%-12s %s" % (entry["state"] or "unreadable", "; ".join(parts))


def _render_spec(report):
    lines = ["%s%s" % (os.path.relpath(report["file"]),
                       ": " + report["name"] if report["name"] else "")]
    depth = 0
    for entry in report["chain"]:
        mark = "" if depth == 0 else "  " * (depth - 1) + "└ "
        lines.append("  %-34s %s" % (mark + os.path.basename(entry["file"]),
                                     _describe_file(entry)))
        depth += 1

    def below(node, depth):
        for child in report["below"].get(node, []):
            lines.append("  %-34s %s" % ("  " * (depth - 1) + "└ " + os.path.basename(child),
                                         _describe_file(report["files"][child])))
            below(child, depth + 1)
    below(report["file"], depth)
    for leaf, step in report["next"].items():
        prefix = "next" if len(report["next"]) == 1 else \
            "next for %s" % (os.path.basename(leaf),)
        lines.append("%s: %s" % (prefix, step))
    for entry in report["files"].values():
        g = entry["generated"]
        for reason in (g or {}).get("reasons", []):
            lines.append("  %s: %s" % (os.path.basename(entry["file"]), reason))
    return "\n".join(lines)


def _spec_run(args):
    if len(args.inputs) != 1:
        print("pro status: give one spec", file=sys.stderr)
        return 2
    absent = missing(*SPEC_NEEDS)
    if absent:
        print("pro status: a spec needs {0}, which this environment lacks.\n"
              "Install it with: pip install 'ymf[spec]'".format(", ".join(absent)),
              file=sys.stderr)
        return 2
    stdout = json_stdout() if args.json else sys.stdout   # before MPI: see json_stdout
    from proteus import Profiling
    if Profiling.procID is None:          # no proteus_default.log (see pro spec)
        Profiling.procID = 0
    path = args.inputs[0]
    if not os.path.isfile(path):
        print("pro status: %s: no such file" % (path,), file=sys.stderr)
        return 2
    try:
        report = spec_status(path)
    except ValueError as exc:
        print("pro status: %s" % (exc,), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False), file=stdout, flush=True)
    else:
        print(_render_spec(report))
    return 0


def run(args):
    if any(p.endswith(".ymf") for p in args.inputs):
        return _spec_run(args)
    from .run import _check_inputs, _context_string, _load
    error = _check_inputs(args)
    if error:
        print("pro status: " + error, file=sys.stderr)
        return 2
    from proteus import runtime
    comm = runtime.init()
    from proteus import Context, Profiling
    Context.contextOptionsString = _context_string(args)
    so, pList, nList, sList, two_phase = _load(args, Profiling.logEvent)
    from .msh import plan_for
    domain, meshOptions, plan = plan_for(pList, nList)
    lines, next_step = describe(args, domain, meshOptions, plan)
    if comm.isMaster():
        print("\n".join(lines))
        print("next:      " + next_step)
    return 0

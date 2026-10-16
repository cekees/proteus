"""``pro status``: what ``pro run`` would do with a model's mesh, and why.

It loads the model (with the same context options the run would get),
works out the mesh plan, and compares it with the manifest ``pro msh``
or ``pro run`` wrote. It runs no mesh generator and writes no manifest.
Loading a model does run its modules, and those may write files of
their own, such as the domain's .poly.
"""
import os
import sys

from . import parser_kwargs

__all__ = ["add_parser", "run", "describe"]


def add_parser(subparsers):
    p = subparsers.add_parser(
        "status",
        help="show whether the mesh is up to date and what pro run would do",
        description="Show a model's mesh plan, whether its mesh files are "
                    "up to date, and what pro run would do. Runs no mesh "
                    "generator (the model's own modules still run).",
        **parser_kwargs())
    p.add_argument("inputs", nargs="+", metavar="MODEL.py",
                   help="so module, TwoPhaseFlow case module, or p and n modules")
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


def run(args):
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

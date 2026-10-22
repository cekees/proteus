"""``pro shell``: a Python prompt with a model loaded, for runs by hand.

This replaces parun's interactive (``-i``) and batch-file (``-b``) modes.
The namespace holds

* ``so``, ``pList``, ``nList``, ``sList``: the loaded model (if one was
  given), as ordinary objects to inspect and change;
* ``opts``: a :class:`proteus.runtime.RunOptions`, the options the run reads;
* ``run(setup_only=False)``: build the NumericalSolution from what those
  objects hold *now* and solve; returns it. Each call is a new run, named
  ``<so.name>0``, ``<so.name>1``, .... It meshes as parun does, through
  NS_base, without pro run's up-to-date check: the model may change
  between runs;
* ``load(*inputs)``: load another model in their place;
* ``comm``, ``proteus``, ``Context``, ``Profiling``.

``--batch FILE`` runs FILE as Python in that namespace instead of opening
the prompt. It is plain Python, not parun's batch format (blocks ended by
``s`` and the file by ``q``): call ``run()`` where parun had ``s``.

A batch file runs on every rank under mpiexec. The prompt is serial only.
"""
import code
import sys

from . import add_command, command

_PRO = command("shell")


__all__ = ["add_parser", "run", "namespace"]


def add_parser(subparsers):
    p = add_command(
        subparsers, "shell",
        help="Python prompt with a model loaded; run() solves it "
             "(parun -i and -b)",
        description="Open a Python prompt, or run a batch file, with a model "
                    "loaded and a run() that solves it with whatever the "
                    "model objects hold at the time.")
    p.add_argument("inputs", nargs="*", metavar="MODEL.py",
                   help="so module, TwoPhaseFlow case module, or p and n "
                        "modules; optional (load() them later)")
    p.add_argument("-b", "--batch", metavar="FILE",
                   help="run FILE as Python in the shell's namespace, then exit")
    p.add_argument("-C", "--context", metavar="'NAME=VALUE ...'",
                   help="context options, as for pro run")
    p.add_argument("-o", "--context-file", metavar="FILE",
                   help="file of context options; -C entries override it")
    p.add_argument("-P", "--petsc", metavar="'-OPTION VALUE ...'",
                   help="PETSc options; these override -O")
    p.add_argument("-O", "--petsc-file", metavar="FILE",
                   help="file of PETSc options ('#' starts a comment)")
    p.add_argument("-l", "--log-level", type=int, default=1, metavar="N",
                   help="log level; 0 turns the log file off (default 1)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="echo the log to the terminal")
    p.set_defaults(handler=run, pumi=False)
    return p


def namespace(args, comm):
    """The shell's namespace, with the model in ``args.inputs`` loaded."""
    import proteus
    from proteus import Context, Profiling, runtime
    from .run import _load, solve

    ns = {"comm": comm, "proteus": proteus, "Context": Context,
          "Profiling": Profiling, "opts": runtime.RunOptions(),
          "so": None, "pList": None, "nList": None, "sList": None,
          "two_phase": False}
    runs = [0]

    def load(*inputs):
        """Load a model (as pro run would) into so, pList, nList, sList."""
        from .run import _check_inputs
        args.inputs = list(inputs)
        error = _check_inputs(args)
        if error:
            raise ValueError(error)
        model = _load(args, Profiling.logEvent)
        ns.update(zip(("so", "pList", "nList", "sList", "two_phase"), model))

    def run(setup_only=False):
        """Solve the loaded model as it stands; returns the NumericalSolution."""
        if ns["so"] is None:
            raise RuntimeError("no model loaded: load('model_so.py') first")
        solution, status = solve(comm, ns["so"], ns["pList"], ns["nList"],
                                 ns["sList"], ns["opts"], ns["two_phase"],
                                 setup_only=setup_only, run_number=runs[0])
        runs[0] += 1
        ns["status"] = status
        return solution

    ns.update(load=load, run=run)
    if args.inputs:
        load(*args.inputs)
    return ns


def run(args):
    from .run import _context_string, _split
    from proteus import runtime
    comm = runtime.init(args.petsc, args.petsc_file)
    if args.batch is None and comm.size() > 1:
        if comm.isMaster():
            print(_PRO + ": the prompt is serial; under mpiexec give a "
                  "batch file with --batch", file=sys.stderr)
        return 2

    from proteus import Context, Profiling
    Profiling.procID = comm.rank()
    if args.verbose:
        Profiling.verbose = True
    if args.log_level > 0:
        stem = _split(args.inputs[0])[1] if args.inputs else "pro_shell"
        Profiling.openLog(stem + ".log", args.log_level)
    Context.contextOptionsString = _context_string(args)

    ns = namespace(args, comm)
    if args.batch is not None:
        with open(args.batch) as f:
            source = f.read()
        exec(compile(source, args.batch, "exec"), ns)
        return ns.get("status", 0)
    model = ("loaded: " + ns["so"].name) if ns["so"] is not None else "no model loaded"
    code.interact(
        banner=_PRO + " ({0}). run() solves the model as it stands; "
               "opts holds the run options; load(...) loads another.".format(model),
        local=ns, exitmsg="")
    return ns.get("status", 0)

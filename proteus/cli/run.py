"""``pro run``: run a model, as ``parun`` does.

The model is one of

* an so module, which names its p and n modules in ``pnList``;
* a case module defining a ``TwoPhaseFlowProblem`` (parun's
  ``--TwoPhaseFlow -f case.py case.py``);
* a p module and an n module.

The kind is read from the file, not from flags or the argument count.

Differences from parun, all deliberate:

* ``-o`` and ``-C`` combine: the file's context options, then the command
  line's, which win. parun uses ``-C`` and ignores ``-o`` when given both.
* ``-O`` and ``-P`` combine the same way. parun keeps ``-O`` and drops
  ``-P``.
* No interactive (``-i``) or batch-file (``-b``) mode, no runtime viewers
  (``-V``, ``-E``, ``-L``, ``--viewMesh``), no ``-d`` debugger flag
  (``python -m pdb -m proteus.cli run ...`` does that), and no ``--SWEs``.
"""
import ast
import os
import sys

from . import parser_kwargs

__all__ = ["add_parser", "run", "model_kind"]


def add_parser(subparsers):
    p = subparsers.add_parser(
        "run",
        help="run a model (what parun does)",
        description="Run a Proteus model: an so module, a TwoPhaseFlow "
                    "case module, or a p module and an n module.",
        **parser_kwargs())
    p.add_argument("inputs", nargs="+", metavar="MODEL.py",
                   help="so module, TwoPhaseFlow case module, or p and n modules")

    g = p.add_argument_group("model options")
    g.add_argument("-C", "--context", metavar="'NAME=VALUE ...'",
                   help="set context options declared by the model")
    g.add_argument("-o", "--context-file", metavar="FILE",
                   help="file of context options; -C entries override it")
    g.add_argument("--pumi", action="store_true",
                   help="TwoPhaseFlow: attach a PUMI adapt manager "
                        "(parun --genPUMI)")

    g = p.add_argument_group("PETSc")
    g.add_argument("-P", "--petsc", metavar="'-OPTION VALUE ...'",
                   help="PETSc options; these override -O")
    g.add_argument("-O", "--petsc-file", metavar="FILE",
                   help="file of PETSc options ('#' starts a comment)")

    g = p.add_argument_group("logging")
    g.add_argument("-l", "--log-level", type=int, default=1, metavar="N",
                   help="log level; 0 turns the log file off (default 1)")
    g.add_argument("-v", "--verbose", action="store_true",
                   help="echo the log to the terminal")
    g.add_argument("-A", "--log-all-processes", action="store_true",
                   help="every rank writes a log, not only rank 0")
    g.add_argument("-p", "--profile", action="store_true",
                   help="profile set-up and the run with cProfile")

    g = p.add_argument_group("output")
    g.add_argument("-D", "--data-dir", default="", metavar="DIR",
                   help="directory for archives and logs (created if needed)")
    g.add_argument("-T", "--text-archive", action="store_true",
                   help="archive in ASCII text instead of HDF5")
    g.add_argument("-G", "--gather-archive", action="store_true",
                   help="gather the per-rank archives at the end")
    g.add_argument("-s", "--subdomain-archives", action="store_true",
                   help="one archive per subdomain instead of a global one")
    g.add_argument("-c", "--cache-archive", action="store_true",
                   help="cache archive writes")
    g.add_argument("-n", "--no-global-sync", action="store_true",
                   help="do not synchronize archive metadata globally")
    g.add_argument("-S", "--save-dof", action="store_true",
                   help="save the degrees of freedom")
    g.add_argument("-B", "--write-vpp", action="store_true",
                   help="write the velocity post-processor")

    g = p.add_argument_group("restart and mesh")
    g.add_argument("-H", "--hot-start", action="store_true",
                   help="restart from the archive")
    g.add_argument("-t", "--hot-start-time", type=float, default=float("inf"),
                   metavar="T", help="time to restart from")
    g.add_argument("-F", "--partitioned-mesh", action="store_true",
                   help="read a mesh partitioned ahead of time")
    g.add_argument("--setup-only", action="store_true",
                   help="set the model up, then stop without solving")

    p.set_defaults(handler=run)
    return p


def model_kind(path):
    """'so' if the file assigns ``pnList`` at module level, else 'case'.

    Read from the source, so nothing is imported (an import would run the
    model before the context options are set).
    """
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "pnList":
                return "so"
    return "case"


def _fail(message):
    print("pro run: " + message, file=sys.stderr)
    return 2


def _split(path):
    path = os.path.abspath(path)
    return os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]


def _options(args):
    from proteus.runtime import RunOptions
    return RunOptions(
        dataDir=args.data_dir,
        useTextArchive=args.text_archive,
        cacheArchive=args.cache_archive,
        gatherArchive=args.gather_archive,
        subdomainArchives=args.subdomain_archives,
        global_sync=not args.no_global_sync,
        hotStart=args.hot_start,
        hotStartTime=args.hot_start_time,
        writeVPP=args.write_vpp,
        generatePartitionedMeshFromFiles=args.partitioned_mesh,
        save_dof=args.save_dof)


def _context_string(args):
    parts = []
    if args.context_file:
        with open(args.context_file) as f:
            parts.append(f.read())
    if args.context:
        parts.append(args.context)
    return " ".join(parts) if parts else None


def _data_dir(comm, data_dir):
    """Create ``data_dir`` on rank 0 and check every rank can write it.
    Returns an error message, or None."""
    if not data_dir:
        return None
    if comm.isMaster() and not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
        except OSError as e:
            return "cannot create data directory {0}: {1}".format(data_dir, e)
    comm.barrier()
    if not os.access(data_dir, os.W_OK):
        return "rank {0} cannot write data directory {1}".format(
            comm.rank(), data_dir)
    return None


def _load(args, log):
    """Return ``(so, pList, nList, sList, two_phase)``."""
    from proteus import defaults, default_s
    if len(args.inputs) == 2:
        (pdir, pname), (ndir, nname) = map(_split, args.inputs)
        log("Loading p module " + pname + " and n module " + nname)
        so = defaults.System_base()
        p = defaults.load_physics(pname, pdir)
        n = defaults.load_numerics(nname, ndir)
        if p.name is None:
            p.name = pname
        so.pnList = [(pname, nname)]
        so.name = p.name
        so.sList = [default_s]
        for attr in ("systemStepControllerType", "systemStepExact",
                     "tnList", "archiveFlag"):
            if hasattr(n, attr):
                setattr(so, attr, getattr(n, attr))
        return so, [p], [n], so.sList, False

    directory, name = _split(args.inputs[0])
    if model_kind(args.inputs[0]) == "so":
        log("Loading so module " + name)
        so, pList, nList = defaults.load_models(name, directory)
        sList = so.sList or [default_s for _ in pList]
        return so, pList, nList, sList, False

    log("Loading TwoPhaseFlow case module " + name)
    import importlib
    from proteus import Context
    from proteus.TwoPhaseFlow.TwoPhaseFlowProblem import TwoPhaseFlowProblem
    sys.path.insert(0, directory)
    case = importlib.import_module(name)
    Context.setFromModule(case)
    problems = [obj for obj in vars(case).values()
                if isinstance(obj, TwoPhaseFlowProblem)]
    if len(problems) != 1:
        raise ValueError(
            "{0}: expected one TwoPhaseFlowProblem, found {1} (and no "
            "pnList, so it is not an so module either)".format(
                args.inputs[0], len(problems)))
    prob = problems[0]
    if args.petsc_file:
        prob.SystemNumerics.usePETScOptionsFileExternal = True
    prob.initializeAll()
    so = prob.so
    so.name = name
    if args.pumi:
        from proteus.MeshAdaptPUMI import MeshAdapt
        prob.domain.AdaptManager = MeshAdapt.AdaptManager()
    pList = [p for p, n in so.pnList]
    nList = [n for p, n in so.pnList]
    sList = so.sList or [default_s for _ in pList]
    return so, pList, nList, sList, True


def run(args):
    if len(args.inputs) > 2:
        return _fail("give one model module, or a p module and an n module")
    for path in args.inputs:
        if not path.endswith(".py"):
            return _fail(path + ": not a .py file")
        if not os.path.isfile(path):
            return _fail(path + ": no such file")
    if len(args.inputs) == 2 and args.pumi:
        return _fail("--pumi applies to a TwoPhaseFlow case module")

    from proteus import runtime
    comm = runtime.init(args.petsc, args.petsc_file)

    from proteus import Profiling
    error = _data_dir(comm, args.data_dir)
    if error:
        print("pro run: " + error, file=sys.stderr)
        comm.comm.tompi4py().Abort(2)
    Profiling.procID = comm.rank()
    Profiling.logAllProcesses = args.log_all_processes
    if args.verbose:
        Profiling.verbose = True
    if args.data_dir:
        Profiling.logDir = args.data_dir
    stem = _split(args.inputs[0])[1]
    if args.log_level > 0:
        Profiling.openLog(stem + ".log", args.log_level,
                          logLocation=args.data_dir or None)
    log = Profiling.logEvent

    from proteus import Context
    Context.contextOptionsString = _context_string(args)

    so, pList, nList, sList, two_phase = _load(args, log)

    import proteus
    from proteus import NumericalSolution
    Profiling.flushBuffer = sList[0].flushBuffer
    log("Running Proteus version " + proteus.__version__, level=0)
    simFlagsList = []
    for p in pList:
        simFlagsList.append({
            "simulationName": p.name,
            "simulationNameProc": p.name + repr(comm.rank()),
            "dataFile": p.name + repr(comm.rank()) + ".dat",
            "components": list(range(p.coefficients.nc)),
        })

    opts = _options(args)
    runName = so.name + "0"
    dispatch = Profiling.Dispatcher(comm, args.profile)
    ns = dispatch(NumericalSolution.NS_base,
                  (so, pList, nList, sList, opts, simFlagsList, two_phase),
                  {}, runName + "_init_prof")
    status = 0
    if not args.setup_only:
        failed = dispatch(ns.calculateSolution, (runName,), {},
                          runName + "_run_prof")
        # reduce, so that every rank exits with the same status
        if comm.globalMax(int(bool(failed))):
            log("%s failed: system step failed" % so.name)
            status = 1
    log("Completed " + so.name)
    return status

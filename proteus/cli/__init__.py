"""``pro``: the Proteus command line, one command with subcommands.

``pro run`` runs a model, as ``parun`` does. The other commands are being
added one phase at a time; see PRO_CLI_RESEARCH.md in the planning notes.

Each subcommand's module imports what it needs inside its ``run``, never
at module level, so ``pro --help`` and argument errors are answered without
starting MPI or PETSc. That matters on login nodes, where starting MPI may
be slow or forbidden.

Commands that need the front-end extras (``ymf[symbolic]``: sympy,
strictyaml) are listed whether or not those are installed. Without them a
command says which extra it needs, instead of argparse reporting an
unknown command.
"""
import argparse
import importlib.util
import sys

__all__ = ["main", "build_parser", "missing"]

#: subcommand -> module in this package; each module defines
#: ``add_parser(subparsers)`` and ``run(args) -> int``
COMMANDS = {
    "run": "run",
    "msh": "msh",
    "status": "status",
    "shell": "shell",
    "spec": "spec",
    "weak": "weak",
}

#: front-end subcommands -> (help, modules they import); not implemented
#: yet, but listed so the help shows where they will go
FRONTEND = {
    "disc": ("fill a spec's discretizations and numerical parameters",
             ("ymf.symbolic", "sympy")),
}


def missing(*modules):
    """The modules in ``modules`` that cannot be imported, without
    importing them."""
    absent = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except ImportError:  # a parent package is absent
            found = False
        if not found:
            absent.append(name)
    return absent


def parser_kwargs():
    """ArgumentParser options this Python supports beyond 3.12's.

    3.14 suggests the nearest command on a typo ("maybe you meant 'run'?")
    and colours help and errors. Older Pythons get neither and still work.
    """
    if sys.version_info >= (3, 14):
        return {"suggest_on_error": True, "color": True}
    return {}


def _version():
    from proteus import __version__
    return __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pro",
        description="Proteus: set up, mesh and run models.",
        **parser_kwargs())
    parser.add_argument("--version", action="version",
                        version="%(prog)s (proteus) " + _version())
    subparsers = parser.add_subparsers(
        dest="command", metavar="<command>", required=True)
    for name, module in COMMANDS.items():
        importlib.import_module("." + module, __name__).add_parser(subparsers)
    for name, (summary, needs) in FRONTEND.items():
        absent = missing(*needs)
        note = " (needs ymf[symbolic])" if absent else " (not implemented yet)"
        sub = subparsers.add_parser(name, help=summary + note,
                                    description=summary + note,
                                    **parser_kwargs())
        sub.set_defaults(handler=_frontend_stub(name, absent))
    return parser


def _frontend_stub(name, absent):
    def handler(args):
        if absent:
            print("pro {0}: needs {1}, which this environment lacks.\n"
                  "Install the front end with: pip install 'ymf[symbolic]'"
                  .format(name, ", ".join(absent)), file=sys.stderr)
        else:
            print("pro {0}: not implemented yet.".format(name),
                  file=sys.stderr)
        return 2
    return handler


#: what turning a spec into a Proteus problem imports (ymf[symbolic])
SPEC_NEEDS = ("ymf.symbolic", "sympy", "strictyaml")


def _is_spec_run(argv):
    """``pro run`` given a YMF spec or archive rather than Python modules."""
    return argv[:1] == ["run"] and any(
        a.endswith(".ymf") for a in argv[1:] if not a.startswith("-"))


def run_spec(argv):
    """``pro run SPEC.ymf ...``: the YMF runner (:mod:`proteus.cli.ymfrun`),
    which has options of its own (``--cells``, ``--check``, ``--emit-pn``...).
    """
    absent = missing(*SPEC_NEEDS)
    if absent:
        print("pro run: a .ymf spec needs ymf[symbolic] to become a Proteus\n"
              "problem, and this environment lacks {0}.\n"
              "Where the front end is installed, write the problem as Proteus\n"
              "files with\n"
              "    pro run SPEC.ymf --emit-pn\n"
              "and run the <stem>_<digest>_so.py it writes here with pro run,\n"
              "which needs no sympy. Or: pip install 'ymf[symbolic]'"
              .format(", ".join(absent)), file=sys.stderr)
        return 2
    # importing ymfrun pins the BLAS/OpenMP threads; nothing numeric is
    # loaded yet at this point
    from . import ymfrun
    return ymfrun.main(argv, prog="pro run")


def main(argv=None):
    """Entry point for the ``pro`` console script; returns the exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if _is_spec_run(argv):
        return run_spec(argv[1:])
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())

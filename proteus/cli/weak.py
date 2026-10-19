"""``pro weak``: the weak forms Proteus would solve a spec with, written out.

The second step of ``pro spec`` -> ``pro weak`` -> ``pro dis`` -> ``pro
run``. ``pro weak SPEC.ymf --list`` says which formulations Proteus
supports for the spec's strong form, and why the others do not apply. ``pro
weak SPEC.ymf`` writes them to ``<stem>.weak.ymf``, a file that ``extends``
the spec, holds one weak form per formulation, and records that it was
generated (``ymf.generated``), so it can be reviewed on its own and checked
later for a stale input or a hand edit.

Each weak form is written from what Proteus will actually solve: the
strong form's scalar residuals, classified into Proteus's
advection-diffusion-reaction slots (``ymf.symbolic.adr.classify``), and
put in Proteus's weak form,

    ∫Ω ∂m/∂t w dx - ∫Ω (f - a∇φ)·∇w dx + ∫Ω (r + H) w dx = boundary terms,

one line per scalar component, each term in the spec's own notation
(``ymf.symbolic.to_notation``). Terms that do not depend on the unknowns
(sources) are written on the ``linear`` side. A hand-written weak form is
prose that nothing checks; this one is derived, and its digest says from
what.

The formulations: Galerkin always; SUPG/PSPG (Proteus's residual-based
velocity-pressure stabilization) where ``ADRProblem.check_stabilization``
says it would run. An existing ``<stem>.weak.ymf`` is regenerated only if a
tool wrote it and nobody has edited it since; otherwise ``pro weak`` stops,
unless told ``--force``.
"""
import json
import os
import sys

from . import add_command, command, json_stdout, load_error, load_errors, missing

_PRO, _DIS = command("weak"), command("dis")


__all__ = ["add_parser", "run", "formulations", "weak_form", "NEEDS"]

NEEDS = ("ymf.symbolic", "ymf.generated", "sympy", "strictyaml")

#: (label, stabilization method as the spec spells it, name to show)
FORMULATIONS = (
    ("proteus_galerkin", "none", "Galerkin"),
    ("proteus_supg_pspg", "SUPG/PSPG", "SUPG/PSPG (Proteus ASGS)"),
)

_SUPERSCRIPT = {2: "²", 3: "³"}


def add_parser(subparsers):
    p = add_command(
        subparsers, "weak",
        help="write the weak forms Proteus would solve a spec with",
        description="Write the weak forms Proteus supports for a YMF spec's "
                    "strong form to <stem>.weak.ymf, a file that extends the "
                    "spec. --list says which apply, and writes nothing.")
    p.add_argument("spec", metavar="SPEC.ymf", help="a well-posed YMF spec")
    p.add_argument("--list", action="store_true",
                   help="say which formulations apply, and write nothing")
    p.add_argument("--only", nargs="+", metavar="LABEL",
                   help="write only these formulations")
    p.add_argument("-o", "--output", metavar="FILE",
                   help="where to write (default: <stem>.weak.ymf beside the spec)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a file that was edited since, or not generated")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(handler=run)
    return p


# --- what applies -------------------------------------------------------------


def formulations(problem):
    """``[{label, method, name, applies, why}]`` for an ADR problem."""
    from proteus.ADRProblem import check_stabilization
    out = []
    for label, method, name in FORMULATIONS:
        try:
            check_stabilization(problem, method)
            applies, why = True, None
        except ValueError as exc:
            applies, why = False, str(exc)
        out.append({"label": label, "method": method, "name": name,
                    "applies": applies, "why": why})
    return out


# --- writing one weak form ---------------------------------------------------


def _depends(expr, space):
    from ymf.symbolic import D, Dt
    return bool(expr.free_symbols & set(space.component_symbols)) or expr.has(D, Dt)


def _split(expr, space):
    """(terms that depend on the unknowns, terms that do not)."""
    import sympy
    expr = sympy.sympify(expr)
    dep = [t for t in sympy.Add.make_args(expr) if _depends(t, space)]
    src = [t for t in sympy.Add.make_args(expr) if not _depends(t, space)]
    return sympy.Add(*dep), sympy.Add(*src)


def _paren(expr):
    from ymf.symbolic import to_notation
    text = to_notation(expr)
    return "(%s)" % text if expr.is_Add or text.startswith("-") else text


def _vector(vec, space):
    """``(sign, text)`` for a flux vector, as a person would write it.

    A common minus sign comes out (``sign`` -1), and a multiple of one
    unknown's gradient is written ``c ∇φ`` rather than its components --
    ``∇u``, where the components would be ``(∂u/∂x, ∂u/∂y)``. Both forms
    read back as the same vector.
    """
    import sympy
    from ymf.symbolic import D, to_notation
    entries = [sympy.sympify(e) for e in vec]
    sign = 1
    nonzero = [e for e in entries if e != 0]
    if nonzero and all(e.could_extract_minus_sign() for e in nonzero):
        sign, entries = -1, [-e for e in entries]
    # c ∇φ: entry k is c * D(φ, k), with the same c and φ for every k
    first = entries[0]
    gradient = None
    for d in first.atoms(D):
        if d.args[0].is_Symbol and int(d.args[1]) == 0:
            c = sympy.simplify(first / d)
            if not c.has(D) and all(
                    sympy.simplify(entries[k] - c * D(d.args[0], k)) == 0
                    for k in range(space.dim)):
                gradient = (c, d.args[0])
                break
    if gradient is not None:
        c, phi = gradient
        prefix = "" if c == 1 else _paren(c) + " "
        return sign, "%s∇%s" % (prefix, phi.name)
    return sign, to_notation(sympy.Matrix(entries))


def _tests(problem, space):
    """The test function of each scalar component: w, or w_i and q."""
    from proteus.ADRProblem import velocity_pressure
    try:
        velocity, pressure = velocity_pressure(problem)
    except ValueError:
        if len(space.components) == 1:
            return {space.components[0]: "w"}
        return {c: "w_" + c for c in space.components}
    tests = {c: "w_%d" % i for i, c in enumerate(velocity)}
    tests[pressure] = "q"
    return tests


def _lines(space, residuals, tests):
    """``(bilinear lines, linear lines)``, one per scalar component."""
    import sympy
    from ymf.symbolic import D, Dt, to_notation
    from ymf.symbolic.adr import classify
    bilinear, linear = [], []
    for component, residual in zip(space.components, residuals):
        eq = classify(residual, space, where="equation for %s" % component)
        w = tests[component]
        lhs, rhs = [], []
        if eq.mass != 0:
            lhs.append("∫Ω %s %s dx" % (to_notation(Dt(eq.mass)), w))
        flux = list(eq.advection)
        for phi, tensor in eq.diffusion.items():
            grad = [D(sympy.Symbol(phi, real=True), k) for k in range(space.dim)]
            for j in range(space.dim):
                flux[j] -= sum(tensor[j, k] * grad[k] for k in range(space.dim))
        parts = [_split(f, space) for f in flux]
        dep_flux = sympy.Matrix([p[0] for p in parts])
        src_flux = sympy.Matrix([p[1] for p in parts])
        if any(e != 0 for e in dep_flux):
            sign, text = _vector(dep_flux, space)
            lhs.append("%s ∫Ω %s·∇%s dx" % ("+" if sign < 0 else "-", text, w))
        dep, src = _split(eq.reaction + eq.hamiltonian, space)
        if dep != 0:
            lhs.append("+ ∫Ω %s %s dx" % (_paren(dep), w))
        if src != 0:
            rhs.append("∫Ω %s %s dx" % (_paren(-src), w))
        if any(e != 0 for e in src_flux):
            sign, text = _vector(src_flux, space)
            rhs.append("%s ∫Ω %s·∇%s dx" % ("-" if sign < 0 else "+", text, w))
        text = " ".join(lhs) or "0"
        if text.startswith("+ "):
            text = text[2:]
        bilinear.append("%s: %s" % (component, text))
        linear.append("%s: %s" % (component, " ".join(rhs) or "0"))
    return bilinear, linear


def _spaces(problem, space, galerkin_saddle):
    """``(trial, test)`` space names, from the unknowns and their Dirichlet data."""
    from proteus.ADRProblem import velocity_pressure
    try:
        _, pressure = velocity_pressure(problem)
    except ValueError:
        pressure = None
    pinned = {e["component"] for e in problem.get("dirichlet") or []}
    trial, test = [], []
    for name, comps in problem["unknowns"].items():
        power = _SUPERSCRIPT.get(len(comps), "^%d" % len(comps)) if len(comps) > 1 else ""
        if galerkin_saddle and comps == [pressure]:
            trial.append("L2(Ω)")
            test.append("L2(Ω)")
            continue
        trial.append("H1(Ω)" + power)
        test.append(("H1_0(Ω)" if pinned & set(comps) else "H1(Ω)") + power)
    return " × ".join(trial), " × ".join(test)


def weak_form(problem, space, residuals, formulation, versions):
    """The weak form entry for one formulation, as the spec's schema has it."""
    tests = _tests(problem, space)
    bilinear, linear = _lines(space, residuals, tests)
    supg = formulation["method"] == "SUPG/PSPG"
    saddle = "q" in tests.values()
    if supg:
        bilinear.append("+ Σ_K τ_K ∫K R(v, p)·(ρ (v·∇)w + ∇q) dx, with R the momentum "
                        "residual and τ_K Proteus's (SubgridError."
                        "NavierStokesASGS_velocity_pressure)")
    trial, test = _spaces(problem, space, saddle and not supg)
    entry = {
        "label": formulation["label"],
        "provenance": "proteus_derived",
        "derivation": (
            "The strong form's scalar residuals (ymf.symbolic %s), classified into "
            "Proteus's slots: ∂m/∂t + ∇·(f - a∇φ) + r + H = 0. Each is multiplied by "
            "its test function (%s) and integrated over Ω, the flux f - a∇φ by parts "
            "(proteus %s)."
            % (versions["ymf"], ", ".join("%s for %s" % (w, c) for c, w in tests.items()),
               versions["proteus"])),
        "solution_spaces": {"trial": trial, "test": test},
        "bilinear": "\n".join(bilinear),
        "linear": "\n".join(linear),
        "boundary_conditions": (
            "Dirichlet values are imposed strongly, so the test functions vanish "
            "where they are given. A neumann or robin condition supplies its "
            "equation's boundary flux (f - a∇φ)·n; elsewhere the flux is the natural "
            "one (zero diffusive flux, outflow advective flux), and on periodic faces "
            "the fluxes cancel."),
        "stabilization_method": formulation["method"],
        "notes": ("One line per scalar component: the line for c in bilinear equals "
                  "the line for c in linear, for every test function of c. Written "
                  "by " + _PRO + " from what Proteus solves, not by hand."),
    }
    if saddle:
        entry["inf_sup_stability"] = (
            "circumvented by PSPG stabilization" if supg else
            "requires an inf-sup stable pair, such as Taylor-Hood (P2/P1)")
    return entry


# --- the command -----------------------------------------------------------------


def _out_path(spec, output):
    from ymf.closure import archive_stem
    if output:
        return os.path.abspath(output)
    return os.path.join(os.path.dirname(os.path.abspath(spec)),
                        archive_stem(spec) + ".weak.ymf")


def _versions():
    import proteus
    import ymf
    return {"proteus": proteus.__version__, "ymf": ymf.__version__}


def _ambiguous(parent, n_new):
    """Discretizations that name no weak form, once there would be several."""
    model = parent.get("model") or {}
    total = len(model.get("weak_forms") or []) + n_new
    if total <= 1:
        return []
    return [d.get("name") for d in (parent.get("solution_paths") or {}).get("discretizations") or []
            if "from_weak_form" not in d]


def plan(spec, only=None):
    """``(report, parent, problem, space, residuals, chosen)``; writes nothing."""
    from ymf.compose import load_composed
    from ymf.symbolic import adr_problem, strong_form_equations
    parent = load_composed(spec, check_state=False)
    parent.pop("outputs", None)
    report = {"spec": str(spec), "name": (parent.get("model") or {}).get("name"),
              "formulations": [], "written": None, "generated": None, "error": None}
    if parent.get("ymf_state") == "abstract":
        report["error"] = "%s is abstract: extend it into a well-posed spec first" % (spec,)
        return report, None, None, None, None, []
    problem = adr_problem(parent)
    space, residuals = strong_form_equations(parent)
    report["formulations"] = formulations(problem)
    labels = {f["label"] for f in report["formulations"]}
    unknown = sorted(set(only or ()) - labels)
    if unknown:
        report["error"] = "no formulation %s (have %s)" % (", ".join(unknown), ", ".join(sorted(labels)))
        return report, parent, problem, space, residuals, []
    chosen = [f for f in report["formulations"]
              if f["applies"] and (not only or f["label"] in only)]
    refused = [f["label"] for f in report["formulations"]
               if only and f["label"] in only and not f["applies"]]
    if refused:
        report["error"] = "%s does not apply: %s" % (
            refused[0], next(f["why"] for f in report["formulations"] if f["label"] == refused[0]))
    return report, parent, problem, space, residuals, chosen


def write(spec, out, parent, problem, space, residuals, chosen, force=False):
    """Write ``out``; return its generated record. Raises ValueError to refuse."""
    from ymf import generated
    if os.path.exists(out) and not force:
        status = generated.check(out)
        if status.record is None:
            raise ValueError("%s exists and was not written by a tool; give -o, "
                             "or --force to overwrite it" % (out,))
        if status.edited:
            raise ValueError("%s was edited since %s wrote it; it is the editor's "
                             "now. Give -o, or --force to overwrite it"
                             % (out, status.record["by"]))
    names = _ambiguous(parent, len(chosen))
    if names:
        # The spec cannot name a weak form this file adds (it would not
        # validate on its own), so naming them is pro dis's to do.
        raise ValueError(
            "discretization%s %s name%s no weak form, which is ambiguous once there "
            "are several. Write one formulation with --only, or leave the "
            "discretizations to %s, which names the weak form each solves"
            % ("s" if len(names) > 1 else "", ", ".join(names),
               "" if len(names) > 1 else "s", _DIS))
    versions = _versions()
    forms = [weak_form(problem, space, residuals, f, versions) for f in chosen]
    extends = os.path.relpath(os.path.abspath(spec), os.path.dirname(out))
    return generated.write(out, extends,
                           "%s (proteus %s, ymf %s)" % (_PRO, versions["proteus"], versions["ymf"]),
                           "model_without_weak_forms", parent,
                           {"model": {"weak_forms": forms}})


def _render(report):
    lines = ["%s%s" % (report["spec"], ": " + report["name"] if report["name"] else "")]
    for f in report["formulations"]:
        lines.append("  %-19s %-26s %s" % (f["label"], f["name"],
                                           "applies" if f["applies"] else
                                           "does not apply: " + f["why"]))
    if report["written"]:
        lines.append("wrote %s (%d weak form%s)" % (
            report["written"], report["count"], "" if report["count"] == 1 else "s"))
    if report["error"]:
        lines.append(_PRO + ": " + report["error"])
    return "\n".join(lines)


def run(args):
    absent = missing(*NEEDS)
    if absent:
        print(_PRO + ": needs {0}, which this environment lacks.\n"
              "Install the front end with: pip install 'ymf[symbolic]'"
              .format(", ".join(absent)), file=sys.stderr)
        return 2
    # before anything imports mpi4py: see json_stdout
    stdout = json_stdout() if args.json else sys.stdout
    from proteus import Profiling
    if Profiling.procID is None:          # no proteus_default.log (see pro spec)
        Profiling.procID = 0
    try:
        report, parent, problem, space, residuals, chosen = plan(args.spec, args.only)
    except load_errors() as exc:          # the loader's, the schema's, the classifier's
        report = {"spec": args.spec, "name": None, "formulations": [], "written": None,
                  "generated": None, "error": load_error(exc, args.spec)}
        chosen = []
    if not report["error"] and not args.list:
        if not chosen:
            report["error"] = "no formulation applies"
        else:
            out = _out_path(args.spec, args.output)
            try:
                report["generated"] = write(args.spec, out, parent, problem, space,
                                            residuals, chosen, force=args.force)
                report["written"], report["count"] = out, len(chosen)
            except ValueError as exc:
                report["error"] = str(exc)
    status = 1 if report["error"] else 0
    if args.json:
        print(json.dumps(dict(report, exit_status=status), indent=2, ensure_ascii=False),
              file=stdout, flush=True)
    else:
        print(_render(report))
    return status

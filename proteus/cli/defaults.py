"""The numerics ``pro disc`` writes when the spec does not choose them.

``pro disc`` fills a spec's discretizations from the table here. They are
Proteus's choices, so they live in Proteus, not in ymf. Every value is a
starting point a person reviews in the ``<stem>.disc.ymf`` it writes, and
changes there (the file is then theirs), or in a file that extends it.

* **Elements** (by formulation and system):

  ==============================  ===============================  ======================
  formulation                     system                           elements
  ==============================  ===============================  ======================
  Galerkin                        scalar or several scalars        CG1 for every field
  Galerkin                        velocity-pressure                Taylor-Hood: CG2 / CG1
  SUPG/PSPG                       velocity-pressure                equal order: CG1 / CG1
  ==============================  ===============================  ======================

  Galerkin on a velocity-pressure system needs an inf-sup stable pair;
  SUPG/PSPG is what makes the equal-order pair stable.

* **Mesh**: ``CELLS`` (16) cells per side on a box; on any other region,
  ``h`` = the shortest side of its bounding box over ``CELLS``. One level.

* **Solver**:

  - ``type``: ``nonlinear`` with Newton when a coefficient depends
    nonlinearly on the unknowns, ``linear`` otherwise.
  - ``linear_solver``: ``mumps`` up to ``DIRECT_DOFS`` (10⁵) unknowns,
    counted on the finest mesh of the study; ``petsc`` above, with PETSc's
    own default Krylov method (GMRES, block Jacobi with ILU(0)) unless the
    spec gives ``petsc_options``. 512 cells of Kovasznay flow was 24 GB with
    MUMPS; an iterative solve is the only one that scales.
  - Where this PETSc has no MUMPS: ``superlu`` (Proteus's own LU), or
    ``petsc`` for a singular operator, which SuperLU cannot take. The
    discretization's ``discretization_details`` says so.

* **Time**: backward Euler with ``STEPS`` (100) steps over the problem's
  time interval, for a problem that has one and a time derivative.
"""

__all__ = ["CELLS", "DIRECT_DOFS", "STEPS", "elements", "mesh", "estimated_dofs",
           "solver", "time"]

#: cells per side of a box (and the divisor of a region's shortest side for h)
CELLS = 16
#: unknowns up to which the default linear solver is a direct one (MUMPS)
DIRECT_DOFS = 100000
#: backward Euler steps over the problem's time interval
STEPS = 100


def _velocity_pressure(problem):
    from proteus.ADRProblem import velocity_pressure
    try:
        return velocity_pressure(problem)
    except ValueError:
        return None


def elements(problem, method):
    """``(finite_element block, short name)`` for a formulation's method."""
    if _velocity_pressure(problem) is None:
        return {"fields": {"family": "CG", "order": 1}}, "p1"
    if method.strip().lower() == "supg/pspg":
        return ({"velocity": {"family": "CG", "order": 1},
                 "pressure": {"family": "CG", "order": 1}}, "p1p1")
    return ({"velocity": {"family": "CG", "order": 2},
             "pressure": {"family": "CG", "order": 1}}, "taylor_hood")


def mesh(problem):
    """The default ``mesh`` block: cells on a box, h on any other region."""
    domain = problem["domain"]
    if domain.get("region") is None:
        return {"cells": CELLS}
    sides = [u - l for l, u in zip(domain["lower"], domain["upper"])]
    return {"h": float(min(sides)) / CELLS}


def _finest(mesh_block):
    """The finest realization of a mesh block: (cells, h, levels)."""
    def last(value, pick):
        if isinstance(value, (list, tuple)):
            return pick(value)
        return value
    return (last(mesh_block.get("cells"), max), last(mesh_block.get("h"), min),
            last(mesh_block.get("levels", 1), max) or 1)


def estimated_dofs(problem, finite_element, mesh_block):
    """Unknowns on the finest mesh of a study, counted on the bounding box.

    A Lagrange space of order k has (k n + 1) nodes along a side of n
    cells, so a box gets the product over its sides. A region meshed by h
    is counted on its bounding box with n = side / h: an overestimate for a
    region that does not fill its box, which errs towards the scalable
    solver.
    """
    from proteus.cli.ymfrun import element_spaces
    cells, h, levels = _finest(mesh_block)
    refine = 2 ** (int(levels) - 1)
    domain = problem["domain"]
    sides = [u - l for l, u in zip(domain["lower"], domain["upper"])]
    if cells is not None:
        per_side = [int(cells) * refine] * len(sides)
    else:
        per_side = [max(1, round(s / (float(h) / refine))) for s in sides]
    total = 0
    for family, order in element_spaces(problem, finite_element).values():
        nodes = 1
        for n in per_side:
            nodes *= (max(int(order), 1) * n + 1) if family == "CG" else \
                (n * (int(order) + 1))
        total += nodes
    return total


def _nonlinear(problem):
    def flags(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k == "depends_on":
                    yield from v.values()
                else:
                    yield from flags(v)
        elif isinstance(value, list):
            for v in value:
                yield from flags(v)
    return "nonlinear" in set(flags(problem["adr"]["equations"]))


def _has_mumps():
    try:
        from petsc4py import PETSc
        return bool(PETSc.Sys.hasExternalPackage("mumps"))
    except Exception:  # no petsc4py at all: no MUMPS either
        return False


def solver(problem, finite_element, mesh_block):
    """``(solver block, estimated unknowns, note or None)`` for one discretization."""
    block = {"type": "nonlinear", "nonlinear_solver": "newton"} if _nonlinear(problem) \
        else {"type": "linear"}
    dofs = estimated_dofs(problem, finite_element, mesh_block)
    note = None
    if dofs > DIRECT_DOFS:
        block["linear_solver"] = "petsc"
    elif _has_mumps():
        block["linear_solver"] = "mumps"
    elif problem.get("up_to_constant"):
        block["linear_solver"] = "petsc"
        note = ("petsc, not mumps: this PETSc has no MUMPS, and the operator is "
                "singular, which superlu cannot solve")
    else:
        block["linear_solver"] = "superlu"
        note = "superlu, not mumps: this PETSc has no MUMPS"
    return block, dofs, note


def time(problem):
    """The ``discretization`` block for a time-dependent problem, or None."""
    interval = problem.get("time")
    if not interval or not problem.get("has_mass"):
        return None
    t0, t1 = (float(v) for v in interval)
    return {"time_integration": "backward_euler", "dt": (t1 - t0) / STEPS}

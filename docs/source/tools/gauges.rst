.. _gauges:

Gauges
******

source: :py:mod:`proteus.Gauges`

Gauges record the value of a finite element field at points, along lines, or
integrated over lines, the way a probe or wave staff does in a laboratory
experiment.  They are auxiliary variables: add one to the ``auxiliaryVariables``
list of a model and Proteus samples it every time step.  Results are written to
a CSV file by one process, whatever the number of ranks.

Usage
=====

.. code-block:: python

   from proteus.Gauges import PointGauges, LineGauges, LineIntegralGauges

   points = PointGauges(gauges=((('u', 'v'), ((0.5, 0.5, 0), (1.0, 0.5, 0))),
                                (('p',), ((0.5, 0.5, 0),))),
                        activeTime=(0, 2.5),
                        sampleRate=0.2,
                        fileName='points.csv')

   lines = LineGauges(gauges=((('u',), (((0, 0.5, 0), (2.2, 0.5, 0)),)),),
                      fileName='lines.csv')

   column = LineIntegralGauges(gauges=((('vof',), (((1.0, 0, 0), (1.0, 1.0, 0)),)),),
                               fileName='column.csv')

   auxiliaryVariables = [points, lines, column]

Each gauge is a pair: a tuple of field names, and the locations to read those
fields at.  A bare string is one field name.  The field names are the model's
``variableNames`` (for example ``'u'``, ``'v'``, ``'p'`` for ``RANS2P``);
a name the model does not have raises a ``ValueError`` that lists the ones it
has.  Locations always have three coordinates, with ``z = 0`` in 2D.

Point gauges
------------

``PointGauges`` reads the finite element function at each point.  The point is
located once, when the model is attached: the mesh node nearest to the point is
found with a kd-tree, and the elements around that node, then around their
vertices, and so on out to five rings, are searched for one that contains the
point.  The value is then a fixed linear combination of the degrees of freedom
of that element, so reading it costs one sparse matrix-vector product.  A point
that is outside the mesh, or that no search ring finds an element for, reads the
value of its nearest node, and Proteus logs a warning saying so.

.. figure:: /_generated/point_location_rings.svg

   The search for the element containing a point, ring by ring.

Line gauges
-----------

``LineGauges`` reads the field at every place the line crosses an element
boundary, so the output shows the finite element function exactly (it is
piecewise linear along the line for first-order elements).  Each crossing is a
column of the output.

Line integral gauges
--------------------

``LineIntegralGauges`` reports one number per line, the integral of the field
along it.  Within each element the integrand is linear along the line, and the
trapezoid rule over the element crossings is exact for first-order elements.
The integral of a volume fraction along a vertical line is the height of that
phase in the column, which is how an equivalent water height is measured.

.. note::

   Line gauges clip the line against the elements it can touch when the model is
   attached.  A bounding-box test discards the rest in one vectorized step, so a
   line through a mesh of 180,000 elements takes tens of milliseconds, and the
   clipping happens once, not every time step.

Lines and points outside the mesh
---------------------------------

A line that leaves the mesh, or crosses a hole such as a cylinder, is restricted
to the part that lies inside.  Proteus logs a warning with the fraction of the
line that was used.  A line that is entirely outside raises a ``ValueError`` on
every rank.

Gauges are often placed "on the bottom" by coordinates that differ from the
mesh's by roundoff.  Points and line endpoints that lie outside the mesh's
bounding box by no more than :math:`10^{-9}` of its largest extent are moved onto
the box.  Only boundaries aligned with the box are snapped: a gauge on a curved or
tilted boundary that misses it by roundoff is restricted (lines) or read at the
nearest node (points).

.. figure:: /_generated/line_gauge_cases.svg

   The four cases of a line gauge: points at element crossings, a line a roundoff
   outside the mesh that is snapped onto it, a line partly outside that is
   restricted, and a line across a hole.

Sampling
--------

With ``sampleRate=0`` (the default) a gauge samples every time step.  Otherwise
samples are anchored to the grid :math:`t_0 + k\,\mathrm{sampleRate}`, where
:math:`t_0` is the first sample, and each is taken at the first time step at or
after its grid time.  The mean interval is therefore the ``sampleRate``, and
single intervals can be up to one time step longer.  A ``sampleRate`` below the
time step samples every step.  ``activeTime=(start, end)`` restricts sampling to
that interval.

Output
------

The CSV file has a header with the field and location of every column, then one
row per sample, starting with the time.  After an h-adapt the file is continued, without a second header.  The header is written once, with
coordinates to six digits.

Parallel runs
-------------

Every rank searches its own elements for each gauge.  The rank that owns a point
is the highest-numbered rank that finds a containing element, or, when none does,
the rank with the nearest node.  The owners send their values to one rank, which
writes the file.  The gauge file is the same for any number of ranks, up to
roundoff.

Moving and adapted meshes
-------------------------

Gauges are located when the model is attached, so on an ALE or r-adaptive mesh a
gauge follows the mesh instead of staying at a fixed point in space.  Under
h-adaptivity with PUMI the gauges are rebuilt after each adapt.

Reference
=========

.. automodule:: proteus.Gauges
   :members: PointGauges, LineGauges, LineIntegralGauges, Gauges

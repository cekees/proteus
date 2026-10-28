"""Sphinx extension that draws the figures used in proteus docstrings.

The figures are generated from seeded code at build time, so they cannot
drift from the algorithm they illustrate.  Each figure is written to
``_generated/`` under the documentation source directory; docstrings refer
to it as ``/_generated/<name>.svg``.
"""
import os

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def point_location_layers(nlayers=3, n=13, seed=11):
    """Returns (points, triangles, nearest node, layers) for the point
    location search on a jittered unstructured 2D mesh.  Each layer is
    (support nodes, new elements, all elements searched so far): the
    support nodes of layer k+1 are the vertices of the elements of layer k
    that were not support nodes before."""
    rng = np.random.default_rng(seed)
    grid = np.array([(i/(n-1), j/(n-1)) for j in range(n) for i in range(n)])
    h = 1/(n-1)
    interior = (grid > 0) & (grid < 1)
    pts = grid + np.where(interior, rng.uniform(-.32*h, .32*h, grid.shape), 0)
    tri = Delaunay(pts).simplices
    nearest = int(cKDTree(pts).query(np.array([0.47, 0.52]))[1])
    node_elements = {}
    for e, t in enumerate(tri):
        for v in t:
            node_elements.setdefault(int(v), set()).add(e)
    seen_elements, seen_nodes, layers = set(), set(), []
    support = {nearest}
    for _ in range(nlayers):
        elements = set()
        for v in support:
            elements |= node_elements[v]
        new_elements = elements - seen_elements
        seen_elements |= elements
        vertices = set()
        for e in elements:
            vertices |= set(map(int, tri[e]))
        seen_nodes |= support
        layers.append((set(support), new_elements, set(seen_elements)))
        support = vertices - seen_nodes
    return pts, tri, nearest, layers


def point_location_rings(nlayers=3):
    """Returns the SVG text of the point location search, one panel per
    layer: red nodes are searched through their elements (black edges),
    their new neighbours (black dots) support the next layer, and grey
    edges were searched already."""
    pts, tri, nearest, layers = point_location_layers(nlayers)
    W, pad = 300, 14

    def X(p):
        return pad + p[0]*(W - 2*pad)

    def Y(p):
        return pad + (1 - p[1])*(W - 2*pad)

    titles = ["layer 1: star of the nearest node",
              "layer 2: elements of the support nodes",
              "layer 3: elements of the new support nodes"]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'font-family="Helvetica,Arial,sans-serif" font-size="13">'
           % (nlayers*W + 20, W + 58),
           '<rect width="100%" height="100%" fill="white"/>']
    for k, (support, new_elements, searched) in enumerate(layers):
        out.append('<g transform="translate(%d,30)">' % (k*(W + 10)))
        out.append('<text x="%g" y="-12" text-anchor="middle" font-weight="bold">%s</text>'
                   % (W/2, titles[k]))
        out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#ddd"/>'
                   % (pad, pad, W - 2*pad, W - 2*pad))
        for e, t in enumerate(tri):
            poly = " ".join("%.1f,%.1f" % (X(pts[v]), Y(pts[v])) for v in t)
            if e in new_elements:
                style = 'fill="#eee" stroke="black" stroke-width="1.6"'
            elif e in searched:
                style = 'fill="none" stroke="#999" stroke-width="1"'
            else:
                style = 'fill="none" stroke="#e4e4e4" stroke-width="0.6"'
            out.append('<polygon points="%s" %s/>' % (poly, style))
        neighbours = set()
        for e in new_elements:
            neighbours |= set(map(int, tri[e]))
        for v in neighbours - support:
            out.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="black"/>' % (X(pts[v]), Y(pts[v])))
        for v in support:
            out.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="red"/>'
                       % (X(pts[v]), Y(pts[v]), 4.6 if v == nearest else 3.6))
        out.append('<text x="%d" y="%d" font-size="11" fill="#444">%d support node(s), '
                   '%d new element(s)</text>' % (pad, W - 2, len(support), len(new_elements)))
        out.append('</g>')
    out.append('<text x="%g" y="%d" text-anchor="middle" font-size="11" fill="#444">'
               'red: nodes whose elements are searched in this layer; black dots: their new '
               'neighbours (support nodes of the next layer); black edges: elements searched; '
               'grey: already searched</text>' % ((nlayers*W + 20)/2, W + 50))
    out.append('</svg>')
    return "\n".join(out)


def _clip_line_to_triangle(a, b, tri):
    """Parameter interval [t0, t1] of the segment a + t (b - a) inside the
    triangle (Cyrus-Beck against its three half-planes), or None."""
    d = b - a
    t0, t1 = 0., 1.
    center = tri.mean(axis=0)
    for i in range(3):
        p, q = tri[i], tri[(i + 1) % 3]
        edge = q - p
        normal = np.array([edge[1], -edge[0]])
        if normal.dot(center - p) > 0:
            normal = -normal  # outward
        num = normal.dot(p - a)
        den = normal.dot(d)
        if abs(den) < 1e-14:
            if num < -1e-14:
                return None
            continue
        t = num / den
        if den > 0:
            t1 = min(t1, t)
        else:
            t0 = max(t0, t)
        if t0 > t1 + 1e-12:
            return None
    return t0, t1


def _turek_mesh():
    """Nodes and triangles of the stored flow-around-a-cylinder mesh used by
    the gauge tests, or None when the test directory is not available."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                        'test', 'cylinder2D', 'conforming_rans2p', 'mesh_cylinder2D')
    try:
        def rows(suffix):
            with open(base + suffix) as f:
                lines = [l.split() for l in f if l.strip() and not l.lstrip().startswith('#')]
            return lines[1:]
        nodes = {int(r[0]): (float(r[1]), float(r[2])) for r in rows('.node')}
        ids = sorted(nodes)
        index = {k: i for i, k in enumerate(ids)}
        pts = np.array([nodes[k] for k in ids])
        tris = np.array([[index[int(v)] for v in r[1:4]] for r in rows('.ele')])
    except (IOError, OSError):
        return None
    return pts, tris


def line_gauge_cases():
    """Returns the SVG text of four line-gauge cases on a jittered 2D mesh:
    the points a line gauge places, a line a roundoff outside the mesh
    that is snapped onto it, a line partly outside that is restricted to
    the mesh, and a line crossing a hole in the mesh."""
    rng = np.random.default_rng(5)
    n = 11
    grid = np.array([(i/(n-1), j/(n-1)) for j in range(n) for i in range(n)])
    h = 1/(n-1)
    interior = (grid > 0) & (grid < 1)
    pts = grid + np.where(interior, rng.uniform(-.3*h, .3*h, grid.shape), 0)
    tri_all = Delaunay(pts).simplices
    centroids = pts[tri_all].mean(axis=1)
    hole_center, hole_radius = np.array([0.5, 0.5]), 0.17
    keep = np.linalg.norm(centroids - hole_center, axis=1) > hole_radius
    W, pad = 250, 30

    state = {'pts': pts, 'x0': 0., 'y0': 0., 'scale': W - 2*pad}

    def X(p):
        return pad + (p[0] - state['x0'])*state['scale']

    def Y(p):
        return pad + (state['ytop'] - p[1])*state['scale']

    def crossings(a, b, tris):
        intervals = []
        for t in tris:
            c = _clip_line_to_triangle(a, b, state['pts'][t])
            if c is not None and c[1] - c[0] > 1e-9:
                intervals.append(c)
        intervals.sort()
        merged = []
        for t0, t1 in intervals:
            if merged and t0 <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], t1)
            else:
                merged.append([t0, t1])
        return merged

    def at(a, b, t):
        return a + t*(b - a)

    def element_ticks(a, b, tris):
        ts = {0.0}
        for t in tris:
            c = _clip_line_to_triangle(a, b, state['pts'][t])
            if c is not None and c[1] - c[0] > 1e-9:
                ts.update(c)
        return sorted(ts)

    turek = _turek_mesh()
    cases = [
        ("1. points at element crossings", np.array([0.04, 0.12]), np.array([0.96, 0.30]), None, False),
        ("2. roundoff outside: snapped", np.array([0.04, -0.05]), np.array([0.96, -0.05]), 0.0, False),
        ("3. partly outside: restricted", np.array([-0.12, 0.78]), np.array([0.70, 0.90]), None, False),
        ("4. hole (cylinder mesh): gap skipped", np.array([0.04, 0.50]), np.array([0.96, 0.50]), None, True),
    ]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'font-family="Helvetica,Arial,sans-serif" font-size="12">' % (4*W, W + 60),
           '<rect width="100%" height="100%" fill="white"/>']
    for k, (title, a, b, snap_y, holed) in enumerate(cases):
        tris = tri_all[keep] if holed else tri_all
        if holed and turek is not None:
            tpts, ttri = turek
            state.update(pts=tpts, x0=-0.02, y0=0., ytop=0.41 + 0.07, scale=(W - 2*pad)/0.54)
            ttri = ttri[tpts[ttri][:, :, 0].min(axis=1) < 0.52]
            tris = ttri
            a, b = np.array([-1e-12, 0.2]), np.array([0.52, 0.2])
        else:
            state.update(pts=pts, x0=0., y0=0., ytop=1., scale=W - 2*pad)
        pts_k = state['pts']
        out.append('<g transform="translate(%d,28)">' % (k*W))
        out.append('<text x="%g" y="-10" text-anchor="middle" font-weight="bold">%s</text>' % (W/2, title))
        out.append('<clipPath id="panel%d"><rect x="%d" y="%d" width="%d" height="%d"/></clipPath>'
                   % (k, pad, pad, W - 2*pad, W - 2*pad))
        out.append('<g clip-path="url(#panel%d)">' % k)
        for t in tris:
            poly = " ".join("%.1f,%.1f" % (X(pts_k[v]), Y(pts_k[v])) for v in t)
            out.append('<polygon points="%s" fill="#f3f3f3" stroke="#bbb" stroke-width="0.8"/>' % poly)
        out.append('</g>')
        if holed and turek is None:
            out.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="white" stroke="#bbb" stroke-dasharray="3 3"/>'
                       % (X(hole_center), Y(hole_center), hole_radius*(W - 2*pad)))
        used_a, used_b = a, b
        if snap_y is not None:
            # the drawn line is the one the user gave; the gauge uses the snapped one
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#888" stroke-width="1.5" '
                       'stroke-dasharray="5 4"/>' % (X(a), Y(a), X(b), Y(b)))
            used_a, used_b = np.array([a[0], snap_y]), np.array([b[0], snap_y])
            out.append('<text x="%.1f" y="%.1f" font-size="11" fill="#555">given (1e-15 outside, drawn exaggerated)</text>'
                       % (X(np.array([0.04, 0])), Y(np.array([0.04, 0])) + 30))
        merged = crossings(used_a, used_b, tris)
        full = used_a, used_b
        # parts of the line that are not gauged
        cursor = 0.0
        pieces_out = []
        for t0, t1 in merged:
            if t0 > cursor + 1e-9:
                pieces_out.append((cursor, t0))
            cursor = t1
        if cursor < 1 - 1e-9:
            pieces_out.append((cursor, 1.0))
        for t0, t1 in pieces_out:
            p, q = at(*full, t0), at(*full, t1)
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c33" stroke-width="1.6" '
                       'stroke-dasharray="5 4"/>' % (X(p), Y(p), X(q), Y(q)))
        for t0, t1 in merged:
            p, q = at(*full, t0), at(*full, t1)
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="black" stroke-width="2.2"/>'
                       % (X(p), Y(p), X(q), Y(q)))
        for t in element_ticks(*full, tris):
            if any(t0 - 1e-9 <= t <= t1 + 1e-9 for t0, t1 in merged):
                p = at(*full, t)
                out.append('<circle cx="%.1f" cy="%.1f" r="3.4" fill="red"/>' % (X(p), Y(p)))
        if holed and turek is not None:
            c = np.array([0.2, 0.2])
            out.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-size="11" fill="#555">cylinder</text>'
                       % (X(c), Y(c) + 26))
        out.append('</g>')
    notes = ["red dots: gauge points (line gauge columns; line integral uses the segments between them)",
             "solid black: gauged; dashed red: not gauged (outside the mesh, or over the hole), with a warning"]
    for i, text in enumerate(notes):
        out.append('<text x="%g" y="%d" text-anchor="middle" font-size="11" fill="#444">%s</text>'
                   % (2*W, W + 44 + 14*i, text))
    out.append('</svg>')
    return "\n".join(out)


FIGURES = {'point_location_rings.svg': point_location_rings,
           'line_gauge_cases.svg': line_gauge_cases}


def generate(app):
    outdir = os.path.join(app.srcdir, '_generated')
    os.makedirs(outdir, exist_ok=True)
    for name, make in FIGURES.items():
        with open(os.path.join(outdir, name), 'w') as f:
            f.write(make())


def setup(app):
    app.connect('builder-inited', generate)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}

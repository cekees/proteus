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


FIGURES = {'point_location_rings.svg': point_location_rings}


def generate(app):
    outdir = os.path.join(app.srcdir, '_generated')
    os.makedirs(outdir, exist_ok=True)
    for name, make in FIGURES.items():
        with open(os.path.join(outdir, name), 'w') as f:
            f.write(make())


def setup(app):
    app.connect('builder-inited', generate)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}

"""Each analytical solution against its own equation, by finite differences, and Python against C.

test_analyticalSolutions.py pins values the code returned when it was written, which cannot tell a right
answer from a wrong one (it froze the wrong sign of LinearAD_DiracIC.duOfXT, and with it the diffusive and
total fluxes of that family, for years). These tests check what the solutions are supposed to be: the PDE
residual, du against a difference quotient of u, fluxes against their definitions, boundary values, and the
independent C implementations in canalyticalSolutions against the Python ones. Differences are central, with
tolerances set about two orders above what the correct solutions give.
"""
import numpy as np
import pytest
from math import pi

from proteus import AnalyticalSolutions as AS
from proteus import canalyticalSolutions as C

rng = np.random.default_rng(11)
E3 = np.eye(3)


def grad(f, x, h=1e-5):
    return np.array([(f(x + h * E3[i]) - f(x - h * E3[i])) / (2 * h) for i in range(3)])


def div(F, x, n=3, h=1e-5):
    return sum((F(x + h * E3[i])[i] - F(x - h * E3[i])[i]) / (2 * h) for i in range(n))


# ---- steady advection-diffusion-reaction

def test_linear_adr_sine_satisfies_its_equation():
    b = np.array([0.7, -0.3, 0.4])
    a = np.array([[2e-2, 3e-3, 0], [3e-3, 1e-2, 1e-3], [0, 1e-3, 3e-2]])
    s = AS.LinearADR_Sine(b=b, a=a, c=1.3, omega=np.array([2.0, 1.0, -1.5]), omega0=0.4)
    for _ in range(5):
        x = rng.uniform(0, 1, 3)
        assert abs(div(lambda y: s.advectiveFluxOfX(y) + s.diffusiveFluxOfX(y), x) + s.rOfUX(s.uOfX(x), x)) < 1e-6
        np.testing.assert_allclose(s.duOfX(x), grad(s.uOfX, x), atol=1e-6)
        np.testing.assert_allclose(s.totalFluxOfX(x), s.advectiveFluxOfX(x) + s.diffusiveFluxOfX(x), atol=1e-14)
    assert abs((s.rOfUX(0.5, x) - s.rOfUX(0.0, x)) / 0.5 - s.drOfUX(0.5, x)) < 1e-12


@pytest.mark.parametrize('b,a', [(1.0, 0.5), (5.0, 0.1), (0.0, 0.3)])
def test_linear_ad_steady_state(b, a):
    s = AS.LinearAD_SteadyState(b, a)
    assert abs(s.uOfX([0.0]) - 1.0) < 1e-12 and abs(s.uOfX([1.0])) < 1e-12
    flux = lambda y: b * s.uOfX([y]) - a * (s.uOfX([y + 1e-5]) - s.uOfX([y - 1e-5])) / 2e-5
    for x in (0.2, 0.5, 0.8):
        assert abs((flux(x + 1e-4) - flux(x - 1e-4)) / 2e-4) < 1e-4 * max(b, 1.0)


@pytest.mark.parametrize('q,r', [(2, 1), (1, 2)])
def test_nonlinear_ad_steady_state(q, r):
    b, a = 1.0, 0.5
    s = AS.NonlinearAD_SteadyState(b=b, q=q, a=a, r=r)
    u = lambda y: s.uOfX([y])
    assert abs(u(0.0) - 1.0) < 1e-6 and abs(u(1.0)) < 1e-6
    flux = lambda y: b * u(y) ** q - a * (u(y + 1e-5) ** r - u(y - 1e-5) ** r) / 2e-5
    for x in (0.2, 0.5, 0.8):
        assert abs((flux(x + 1e-3) - flux(x - 1e-3)) / 2e-3) < 2e-3


@pytest.mark.parametrize('nd', [1, 2, 3])
def test_poissons_equation_source_is_the_laplacian(nd):
    # r is what proteus's reaction term needs: -div(grad u) - f = 0 with r = -f, so laplacian(u) = r
    s = AS.PoissonsEquation(K=10.0, nd=nd)
    for _ in range(4):
        x = rng.uniform(0.1, 0.9, 3)
        lap = sum((s.uOfX(x + 1e-4 * E3[i]) - 2 * s.uOfX(x) + s.uOfX(x - 1e-4 * E3[i])) / 1e-8 for i in range(nd))
        assert abs(lap - s.rOfUX(s.uOfX(x), x)) < 1e-5 * max(abs(lap), 1.0)
    assert abs(s.uOfX(np.array([0.0, 0.3, 0.4]))) < 1e-14 and abs(s.uOfX(np.array([1.0, 0.3, 0.4]))) < 1e-14


# ---- time-dependent advection-diffusion with a Dirac initial condition

@pytest.mark.parametrize('n,b', [(1, [1.0]), (2, [0.6, -0.4]), (3, [0.5, 0.2, -0.3])])
def test_dirac_ic_satisfies_the_advection_diffusion_equation(n, b):
    a = 0.02
    s = AS.LinearAD_DiracIC(n=n, b=np.array(b), a=a, tStart=0.3, u0=0.7, x0=np.array([0.1, 0.0, 0.0]))
    # its own generator: the module's is shared, so where it stands depends on which tests this xdist worker
    # ran before, and with 40 draws a 3-D peak this narrow left only 4 above the cutoff on one CI run
    local = np.random.default_rng(1000 + n)
    checked = 0
    for _ in range(4000):
        if checked == 8:
            break
        x = np.zeros(3); x[:n] = local.uniform(-0.2, 1.0, n); T = local.uniform(0.1, 0.4)
        u = s.uOfXT(x, T)
        if u < 1e-3:
            continue
        checked += 1
        ut = (s.uOfXT(x, T + 1e-5) - s.uOfXT(x, T - 1e-5)) / 2e-5
        total = lambda y: s.advectiveFluxOfXT(y, T) + s.diffusiveFluxOfXT(y, T)
        assert abs(ut + div(total, x, n)) < 1e-4 * max(abs(ut), 1.0)
        g = grad(lambda y: s.uOfXT(y, T), x)[:n]
        np.testing.assert_allclose(s.duOfXT(x, T)[:n], g, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(s.diffusiveFluxOfXT(x, T)[:n], -a * g, rtol=1e-4, atol=1e-7)
        np.testing.assert_allclose(s.totalFluxOfXT(x, T), s.advectiveFluxOfXT(x, T) + s.diffusiveFluxOfXT(x, T), atol=1e-14)
    assert checked >= 5


def test_dirac_ic_conserves_its_mass():
    s = AS.LinearAD_DiracIC(n=1, b=np.array([1.0]), a=0.02, tStart=0.0, u0=0.7, x0=np.zeros(3))
    xs = np.linspace(-3, 6, 20001)
    for T in (0.2, 0.5):
        assert abs(np.trapezoid([s.uOfXT(np.array([x, 0, 0]), T) for x in xs], xs) - 0.7) < 1e-6


def test_linear_decay_dirac_ic_satisfies_its_equation():
    a, c = 0.02, 1.3
    s = AS.LinearADR_Decay_DiracIC(n=1, b=np.array([0.8]), a=a, c=c, tStart=0.3, u0=0.7, x0=np.zeros(3))
    for x0 in (0.35, 0.5, 0.7):
        x = np.array([x0, 0, 0]); T = 0.2
        ut = (s.uOfXT(x, T + 1e-5) - s.uOfXT(x, T - 1e-5)) / 2e-5
        total = lambda y: s.advectiveFluxOfXT(y, T) + s.diffusiveFluxOfXT(y, T)
        resid = ut + div(total, x, 1) + s.rOfUXT(s.uOfXT(x, T), x, T)
        assert abs(resid) < 1e-4 * max(abs(ut), 1.0)


# ---- the independent C implementations

def _dirac_rwork(b, n, a, tStart, u0, x0, c=1.3):
    return np.array(list(b) + [n, a, tStart, u0] + list(x0) + [c])


def test_dirac_ic_python_agrees_with_c_and_both_have_du_right():
    b = np.array([0.8, -0.3, 0.2]); n, a, tStart, u0 = 2.0, 0.02, 0.3, 0.7; x0 = np.array([0.1, 0.0, 0.0])
    rwork = _dirac_rwork(b, n, a, tStart, u0, x0); iwork = np.zeros(1, 'i'); T = 0.2
    pts = np.array([[0.3 + 0.1 * i, 0.05 * i, -0.02 * i] for i in range(6)])
    py = AS.LinearAD_DiracIC(n=n, b=b, a=a, tStart=tStart, u0=u0, x0=x0)
    u = np.zeros(len(pts)); C.LinearAD_DiracIC(iwork, rwork, T, pts, u)
    du = np.zeros((len(pts), 3)); C.LinearAD_DiracIC_du(iwork, rwork, T, pts, du)
    fa = np.zeros((len(pts), 3)); C.LinearAD_DiracIC_advectiveVelocity(iwork, rwork, T, pts, fa)
    fd = np.zeros((len(pts), 3)); C.LinearAD_DiracIC_diffusiveVelocity(iwork, rwork, T, pts, fd)
    np.testing.assert_allclose([py.uOfXT(p, T) for p in pts], u, rtol=1e-13)
    fdu = np.array([grad(lambda y: py.uOfXT(y, T), p, 1e-6) for p in pts])
    np.testing.assert_allclose(du, fdu, rtol=1e-5, atol=1e-8)                    # C
    np.testing.assert_allclose([py.duOfXT(p, T) for p in pts], fdu, rtol=1e-5, atol=1e-8)   # python
    np.testing.assert_allclose([py.advectiveFluxOfXT(p, T) for p in pts], fa, rtol=1e-12)
    np.testing.assert_allclose([py.diffusiveFluxOfXT(p, T) for p in pts], fd, rtol=1e-5, atol=1e-8)


# ---- ODE

@pytest.mark.parametrize('a,p', [(5.0, 1.0), (5.0, 2.0), (2.0, 0.5), (5.0, -0.25)])
def test_nonlinear_dae(a, p):
    s = AS.NonlinearDAE(a, p)
    assert s.uOfT(0.0) == pytest.approx(1.0)
    for t in (0.02, 0.05, 0.1):
        ut = (s.uOfT(t + 1e-6) - s.uOfT(t - 1e-6)) / 2e-6
        assert ut == pytest.approx(s.fOfUT(s.uOfT(t), t), rel=1e-5)
    assert s.uOfXT([0.0, 0.0, 0.0], 0.1) == s.uOfT(0.1)       # the AS_base interface works


# ---- flows

def test_plane_couette_flow():
    s = AS.PlaneCouetteFlow_u(plateSeperation=2.0, upperPlateVelocity=0.3, origin=[0.0, 0.5, 0.0])
    ys = np.array([0.5, 1.0, 1.5, 2.5])
    np.testing.assert_allclose([s.uOfX(np.array([0.2, y, 0.1])) for y in ys], 0.3 * (ys - 0.5) / 2.0, atol=1e-14)


@pytest.mark.parametrize('mu,gp,h', [(1.0, 1.0, 1.0), (0.7, -2.0, 1.5)])
def test_plane_poiseuille_flow(mu, gp, h):
    s = AS.PlanePoiseuilleFlow_u(plateSeperation=h, mu=mu, grad_p=gp, origin=[0.0, 0.2, 0.0])
    f = lambda y: s.uOfX(np.array([0.1, y, 0.0]))
    assert abs(f(0.2)) < 1e-12 and abs(f(0.2 + h)) < 1e-12
    for y in (0.3, 0.7, 1.0):
        assert abs(mu * (f(y + 1e-3) - 2 * f(y) + f(y - 1e-3)) / 1e-6 - gp) < 1e-5 * max(abs(gp), 1.0)
    assert AS.PlanePoiseuilleFlow_p(plateSeperation=h, mu=mu, grad_p=gp).uOfX(np.array([0.7, 0, 0])) == pytest.approx(0.7 * gp)


@pytest.mark.parametrize('theta,phi', [(0.0, pi / 2.0), (0.3, 0.9), (1.0, 1.2), (2.0, 0.5), (-0.7, 1.4)])
def test_rotated_plane_poiseuille_flow(theta, phi):
    kw = dict(plane_theta=theta, plane_phi=phi, v_theta=0.4, v_norm=0.8, mu=0.9, grad_p=1.7)
    if phi == pi / 2.0:
        kw.pop('v_theta')                                       # the default flow direction must work
    fields = [AS.PlanePoiseuilleFlow_u2(**kw), AS.PlanePoiseuilleFlow_v2(**kw), AS.PlanePoiseuilleFlow_w2(**kw)]
    p = AS.PlanePoiseuilleFlow_p2(**kw)
    assert abs(np.dot(fields[0].v_n, fields[0].plane_n)) < 1e-12
    x = rng.uniform(0.1, 0.9, 3); h = 1e-4
    vel = lambda y: np.array([f.uOfX(y) for f in fields])
    assert abs(sum((vel(x + h * E3[i])[i] - vel(x - h * E3[i])[i]) / (2 * h) for i in range(3))) < 1e-6
    lap = sum((vel(x + h * E3[i]) - 2 * vel(x) + vel(x - h * E3[i])) / h ** 2 for i in range(3))
    np.testing.assert_allclose(0.9 * lap, grad(p.uOfX, x, h), atol=1e-3)


@pytest.mark.parametrize('n,Re', [(2, 100.0), (1, 10.0)])
def test_vortex_decay_is_a_navier_stokes_solution(n, Re):
    U, V, P = AS.VortexDecay_u(n=n, Re=Re), AS.VortexDecay_v(n=n, Re=Re), AS.VortexDecay_p(n=n, Re=Re)
    nu, h = 1.0 / Re, 1e-4
    f = lambda g, y, t: g.uOfXT(np.array([y[0], y[1], 0.0]), t)
    e = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for _ in range(4):
        x = rng.uniform(0, 1, 2); t = rng.uniform(0.0, 0.5)
        u, v = f(U, x, t), f(V, x, t)
        ut, vt = [(f(g, x, t + 1e-6) - f(g, x, t - 1e-6)) / 2e-6 for g in (U, V)]
        gu = [(f(U, x + h * e[i], t) - f(U, x - h * e[i], t)) / (2 * h) for i in range(2)]
        gv = [(f(V, x + h * e[i], t) - f(V, x - h * e[i], t)) / (2 * h) for i in range(2)]
        gp = [(f(P, x + h * e[i], t) - f(P, x - h * e[i], t)) / (2 * h) for i in range(2)]
        lu = sum((f(U, x + h * e[i], t) - 2 * u + f(U, x - h * e[i], t)) / h ** 2 for i in range(2))
        lv = sum((f(V, x + h * e[i], t) - 2 * v + f(V, x - h * e[i], t)) / h ** 2 for i in range(2))
        scale = (n * pi) ** 2
        assert abs(ut + u * gu[0] + v * gu[1] + gp[0] - nu * lu) < 1e-5 * scale
        assert abs(vt + u * gv[0] + v * gv[1] + gp[1] - nu * lv) < 1e-5 * scale
        assert abs(gu[0] + gv[1]) < 1e-6


# ---- Buckley-Leverett Riemann problem (non-convex flux, rarefaction followed by a shock)

def test_buckley_leverett_riemann_solution():
    alpha = 0.5
    flux = lambda s: s ** 2 / (s ** 2 + alpha * (1 - s) ** 2)

    class Coef(object):
        def initializeElementQuadrature(self, t, c):
            pass

        def evaluate(self, t, c):
            s = c[('u', 0)][0]
            c[('m', 0)][0] = s
            c[('f', 0)][0, 0] = flux(s)
    sol = AS.Buckley_Leverett_RiemannSoln(Coef(), uLeft=1.0, uRight=0.0, t0=0.0, x0=0.0, T=0.5)
    ustar = 1.0 / np.sqrt(3.0)                       # tangent point of the concave envelope: f'(u*) = f(u*)/u*
    speed = flux(ustar) / ustar
    ss = np.linspace(ustar, 1.0, 200001)
    dfds = np.gradient(flux(ss), ss)
    for t in (0.3, 0.5):
        for xi in (-0.5, 0.0, 0.5, 1.0, 1.3, 1.5, 2.5):
            u = sol.uOfXT(np.array([xi * t, 0, 0]), t)
            if xi >= speed:
                expected = 0.0
            elif xi <= dfds[-1]:
                expected = 1.0
            else:
                expected = ss[np.argmin(np.abs(dfds - xi))]
            assert u == pytest.approx(expected, abs=1e-5), (t, xi)


# ---- the vectorized interface of AS_base

def test_vectorized_getters_match_the_point_functions():
    s = AS.LinearAD_DiracIC(n=3, b=np.array([0.6, -0.4, 0.1]), a=0.02, tStart=0.3, u0=0.7, x0=np.array([0.1, 0.0, 0.0]))
    X = np.stack([rng.uniform(0.1, 0.6, (3, 4)), rng.uniform(-0.2, 0.1, (3, 4)), rng.uniform(-0.1, 0.1, (3, 4))], axis=-1)
    T = 0.2
    U = np.zeros((3, 4)); s.getUOfXT(X, T, U)
    DU = np.zeros((3, 4, 3)); s.getDUOfXT(X, T, DU)
    F = np.zeros((3, 4, 3)); s.getDiffusiveFluxOfXT(X, T, F)
    for i in range(3):
        for j in range(4):
            assert U[i, j] == s.uOfXT(X[i, j], T)
            np.testing.assert_array_equal(DU[i, j], s.duOfXT(X[i, j], T))
            np.testing.assert_array_equal(F[i, j], s.diffusiveFluxOfXT(X[i, j], T))
    sine = AS.LinearADR_Sine()
    Y = rng.uniform(0, 1, (5, 3)); R = np.zeros(5); DR = np.zeros(5)
    sine.getROfUX(Y, 0.0, R); sine.getDROfUX(Y, 0.0, DR)
    np.testing.assert_allclose(R, [sine.rOfUX(sine.uOfX(y), y) for y in Y])
    np.testing.assert_allclose(DR, [sine.drOfUX(0.0, y) for y in Y])

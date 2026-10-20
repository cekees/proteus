# Hornung–Messing manufactured problem (2-D, no gravity)

The accuracy test of Ait Hammou Oulhaj, Cancès & Chainais-Hillairet (2018), in the form used by
Dielmann, Kuzmin & Kees (JNMA, in preparation), §7.1:

- Domain Ω = (0,1)², T = 0.05, no gravity (hydraulic head = pressure head p).
- Constitutive laws: s(p) = π²/2 − 2 arctan²(p) and η(p) = 2/(1+p²) for p < 0; s = π²/2 and η = 2 for p ≥ 0.
- `Richards.Coefficients(..., PSK_type='HornungMessing')` with θ_R = 0, θ_S − θ_R = π²/2, K_s = 2.
- Exact solution: p = −ξ/2 for ξ < 0 (saturated) and p = −tan((e^ξ−1)/(e^ξ+1)) for ξ ≥ 0, with ξ = x − y − t.
  It satisfies the PDE on both branches and has continuous flux across ξ = 0.
- Dirichlet data p_ex on ∂Ω, initial data p_ex(·,0).

## Files
- `hm_2d_p.py`, `hm_2d_n.py`: the decks. Context options: `STABILIZATION_TYPE`, `FCT`, `nn`
  (nodes per side), `nDT` (fixed steps), `T`, `theta_time` (1 = backward Euler, 0.5 = Crank–Nicolson,
  through `Richards.ThetaScheme`).
- `hm_l2_error.py`: L² error at t = T. P1 interpolation, Dunavant degree-8 rule on 4 subtriangles per element.
- `run_hm_convergence.sh`: h = 1/2…1/64. θ = 1 uses Δt = 0.2h²; θ = ½ uses Δt = 0.1h, as in the JNMA tables.
  Writes `convergence_theta1.0.txt` and `convergence_theta0.5.txt`. A full serial sweep takes about 3 min.

```bash
./run_hm_convergence.sh
```

## Results (structured right-triangle P1 mesh, serial LU, Newton tolerance 1e-10)

`convergence_theta*.txt` holds the full tables. In summary:
- Low order (`STABILIZATION_TYPE=2 FCT=False`): first order on the finer meshes (1.00 at h = 1/64). It has the
  same non-monotone coarse-mesh regime (error rising from h = 1/2 to 1/4) as the JNMA Table 1.
- Galerkin, θ = 1: about second order (1.9–2.2). Galerkin, θ = ½: about second order (1.8–2.5).
- FCT (WRR scheme, `STABILIZATION_TYPE=2 FCT=True`): **does not converge reliably**. Its error is
  concentrated in the saturated region ξ < 0 (p > 0). After limiting, the water content there sits at or
  just below θ_S, and the θ → p recovery returns p ≈ 0⁻ instead of the positive head. This is the
  non-invertibility of s at saturation that motivates the implicit formulation in the JNMA manuscript.
  The 1-D infiltration benchmarks never enter p > 0, so they do not expose it.

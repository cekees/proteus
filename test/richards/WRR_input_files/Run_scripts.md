# WRR Richards input files

## test_1 -- Celia Fig. 6(b)

1 m column, `nn = 41` (dz = 2.5 cm, exactly Celia's grid), `T = 1 d`,
`nDTout = 1000`; the figure is frame 1000 (24 h).

```bash
cd test_1
./run_test_1.sh                          # stab_0, stab_2, FCT, then the figure
SCHEMES="FCT" ./run_test_1.sh            # one scheme
PLOT_ONLY=1 ./run_test_1.sh              # re-draw from archives on disk
```

Produces `stab_0/`, `stab_2/`, `FCT/` (each with the deck copy, `run.log`, and
`re_vgm_sand_10m_1d.h5`) and `test_1_schemes_vs_celia.png`, plus a
depth-normalized L2 head error per scheme on stdout.



## test_2 -- HYDRUS-1D

20 m column, `nn = 101` (dz = 0.2 m, matching the HYDRUS nodes), `T = 2 d`,
`nDTout = 1000`; the figure overlays frames 104 / 229 / 1000 = 5 / 11 / 48 h.

```bash
cd test_2
./run_test_2.sh                          # stab_2, FCT, then the figure
```

Produces `stab_2/`, `FCT/` and `test_2_schemes_vs_hydrus.png`.

The column wets through at ~26 h and relaxes onto psi = 0 everywhere, so the
48 h frame alone is a weak target -- both schemes match it to round-off (2.7e-16
and 4.0e-17).  The moving front is where the schemes separate: depth-rms psi
1.80 vs 0.56 m at 5 h, 1.00 vs 0.17 m at 11 h, stab_2 vs FCT.  The diagonal
black line in the figure is the 48 h HYDRUS curve, not a scheme.

## test_3 -- Szymkiewicz Fig. 6

5 m column, `nn = 11` (dz = 50 cm), `T = 0.125 d`, `nDTout = 1000`; the figure
is frame 1000 (3 h).

```bash
cd test_3
./run_test_3.sh                          # stab_2, FCT, then the figure
```

Produces `stab_2/`, `FCT/` and `test_3_schemes_vs_szymkiewicz.png`.

Scored against the paper's own dz = 50 cm curve (`K_NEW`, eq. 11 -- 11 points,
one per node), *not* against the converged `K_INT` reference, which is a
dz = 0.05 cm solution: on a 10-cell mesh the front carries ~0.4 m of pure
spatial error either way, so the converged curve measures the grid, not the
scheme.  `K_INT` is still drawn, unscored, to show where both are heading.

## Tracy_convergence -- spatial convergence, 2D

10 x 10 m unstructured domain, Gardner retention (`PSK_type='Gardner'`, alpha =
0.164 1/m), refinements ref_2..ref_6 (nnx 41 / 81 / 161 / 321 / 641, run on
4 / 8 / 16 / 64 / 90 ranks).  This one is MPI-parallel and takes hours.

```bash
cd Tracy_convergence
./run_tracy_convergence.sh                          # 3 schemes x ref_3..ref_6
SCHEMES="FCT" REFS="ref_3 ref_4" ./run_tracy_convergence.sh
ANALYZE_ONLY=1 ./run_tracy_convergence.sh           # skip the runs, re-tabulate

nohup conda run --no-capture-output -n [proteus_env] \
      bash run_tracy_convergence.sh > sweep.log 2>&1 &
```

Each case runs, in `<scheme>/<ref>/`:

```
mpiexec -n <ranks> parun -p re_vgm_sand_10x10m_2d_p.py \
        re_vgm_sand_10x10m_2d_c0p1_n.py -l 5 -v \
        -C "<scheme opts> nnx=<nnx>" \
        -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist"
```

`-p` is proteus's *profile* flag, so each rank leaves `*_init_prof*` and
`*_run_prof*` in the run directory alongside `mpi_run.log`.  Override the solver
with `PETSC_OPTS=`.

### Error

`analyze_convergence_L2.py` is run once per scheme directory at the end, and can
be re-run on its own from inside one:

```bash
cd FCT
REF_MIN=ref_3 REF_MAX=ref_6 T_INDICES=300,350,400 \
    conda run -n quadpy_env python ../analyze_convergence_L2.py
```

It writes `Tracy_L2_convergence.txt` -- `h`, `L2(psi)` and the observed rate `p`
at each output index (300 / 350 / 400 = 3.75e-4 / 4.375e-4 / 5.0e-4 d) -- and
skips refinements whose `.h5` is not there yet, so it is safe to run against a
sweep still in flight.  It is self-contained: the Tracy solution is inlined, so
it needs no `Error_rate.py`.

`analyze_convergence_fast.py` (also here) is the wider table: Linf, mass error,
over/undershoot and wall-clock as well.

## Raingarden -- two-layer rain garden vs HYDRUS-2D/3D

```bash
cd Raingarden
./run_raingarden.sh                       # 3 schemes + both figures
SCHEMES="FCT" ./run_raingarden.sh         # one scheme
NP=2 ./run_raingarden.sh                  # MPI ranks per run
PLOT_ONLY=1 ./run_raingarden.sh           # re-draw from existing archives
PETSC_OPTS="..." ./run_raingarden.sh      # override the linear solver
```

```bash
SCHEME=stab_2 python plot_infiltration_panels.py   # panels for another scheme
python compare_hydrus_vs_schemes.py                # contours vs HYDRUS
python table_from_log.py                           # solver cost, per scheme
```

## bio2d -- two-dimensional bioswale vs HYDRUS-2D


```bash
cd bio2d
./run_bio2d.sh                            # 3 schemes + both figures + both tables
SCHEMES="FCT" ./run_bio2d.sh              # one scheme
HE=0.15 ./run_bio2d.sh                    # finer mesh
MESH=hydrus ./run_bio2d.sh                # solve on the HYDRUS mesh -> hydrus_mesh/
NP=4 ./run_bio2d.sh                       # MPI ranks per run
PLOT_ONLY=1 ./run_bio2d.sh                # re-draw/re-tabulate from what is on disk
PETSC_OPTS="..." ./run_bio2d.sh           # override the linear solver

conda run --no-capture-output -n [Proteu ] bash run_bio2d.sh
```

Each case runs, in `<scheme>/`:

```
mpiexec -n <NP> parun -p Raingarden_p.py Raingarden_n.py -l 5 -v \
        -C "<scheme opts> he=0.3" \
        -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist"
```

]

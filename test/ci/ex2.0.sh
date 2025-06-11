parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=2.0 unstructured=True refinement=1" > ex2.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=2.0 unstructured=True refinement=2" >> ex2.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=2.0 unstructured=True refinement=3" >> ex2.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=2.0 unstructured=True refinement=4" >> ex2.0.out
grep L2 ex2.0.out
grep Linfty ex2.0.out

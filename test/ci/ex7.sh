parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=7.0 unstructured=True refinement=1" > ex7.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=7.0 unstructured=True refinement=2" >> ex7.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=7.0 unstructured=True refinement=3" >> ex7.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=7.0 unstructured=True refinement=4" >> ex7.out
grep L2 ex7.out
grep Linfty ex7.out

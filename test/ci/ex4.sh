parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=4.0 unstructured=True refinement=1" > ex4.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=4.0 unstructured=True refinement=2" >> ex4.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=4.0 unstructured=True refinement=3" >> ex4.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=4.0 unstructured=True refinement=4" >> ex4.out
grep L2 ex4.out
grep Linfty ex4.out

parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=1" > ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=2" >> ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=3" >> ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=4" >> ex1.out
grep L2 ex1.out
grep Linfty ex1.out

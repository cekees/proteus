parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=8.0 unstructured=False refinement=1" > ex8.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=8.0 unstructured=False refinement=2" >> ex8.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=8.0 unstructured=False refinement=3" >> ex8.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=8.0 unstructured=False refinement=4" >> ex8.0.out
grep L2 ex8.0.out
grep Linfty ex8.0.out

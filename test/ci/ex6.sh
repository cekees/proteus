parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=6.0 unstructured=True refinement=1" > ex6.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=6.0 unstructured=True refinement=2" >> ex6.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=6.0 unstructured=True refinement=3" >> ex6.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=6.0 unstructured=True refinement=4" >> ex6.out
grep L2 ex6.out
grep Linfty ex6.out

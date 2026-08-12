parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=1" > ex3.0r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=2" >> ex3.0r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=3" >> ex3.0r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=4" >> ex3.0r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=5" >> ex3.0r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=6" >> ex3.0r.out
grep L2 ex3.0r.out
grep Linfty ex3.0r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=1" > ex3.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=2" >> ex3.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=3" >> ex3.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=4" >> ex3.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=5" >> ex3.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=6" >> ex3.0.out
grep L2 ex3.0.out
grep Linfty ex3.0.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=1" > ex3.0us.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=2" >> ex3.0us.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=3" >> ex3.0us.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=4" >> ex3.0us.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=5" >> ex3.0us.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=6" >> ex3.0us.out
grep L2 ex3.0us.out
grep Linfty ex3.0us.out

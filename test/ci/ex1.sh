parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=1" > ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=2" >> ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=3" >> ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=4" >> ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=5" >> ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True refinement=6" >> ex1.out
grep L2 ex1.out
grep Linfty ex1.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True skew=0.0 refinement=1" > ex1ru.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True skew=0.0 refinement=2" >> ex1ru.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True skew=0.0 refinement=3" >> ex1ru.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True skew=0.0 refinement=4" >> ex1ru.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True skew=0.0 refinement=5" >> ex1ru.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=True skew=0.0 refinement=6" >> ex1ru.out
grep L2 ex1ru.out
grep Linfty ex1ru.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=False refinement=1" > ex1r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=False refinement=2" >> ex1r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=False refinement=3" >> ex1r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=False refinement=4" >> ex1r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=False refinement=5" >> ex1r.out
parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=1.0 unstructured=False refinement=6" >> ex1r.out
grep L2 ex1r.out
grep Linfty ex1r.out

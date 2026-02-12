echo -n "Enter IFEM order to test (1 or 2): "
read num

if [ "$num" -eq 1 ]; then
    echo "Test=3.0 IFEM order 1" > p1.out
    echo "--------------------------------------------------------------"
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=1" > ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=2" >> ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=3" >> ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=4" >> ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=5" >> ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=6" >> ex3.out
    grep L2 ex3.out >> p1.out
    grep Linfty ex3.out >> p1.out
    # echo "--------------------------------------------------------------"
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=1" > ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=2" >> ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=3" >> ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=4" >> ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=5" >> ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=6" >> ex3ru.out
    # grep L2 ex3ru.out
    # grep Linfty ex3ru.out
    # echo "--------------------------------------------------------------"
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=1" > ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=2" >> ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=3" >> ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=4" >> ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=5" >> ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=6" >> ex3r.out
    # grep L2 ex3r.out
    # grep Linfty ex3r.out
elif [ "$num" -eq 2 ]; then
    echo "Test=3.0 IFEM order 2" > p2.out
    echo "--------------------------------------------------------------"
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=1" > ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=2" >> ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=3" >> ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=4" >> ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=5" >> ex3.out
    parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True refinement=6" >> ex3.out
    grep L2 ex3.out >> p2.out
    grep Linfty ex3.out >> p2.out
    # echo "--------------------------------------------------------------"
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=1" > ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=2" >> ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=3" >> ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=4" >> ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=5" >> ex3ru.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=True skew=0.0 refinement=6" >> ex3ru.out
    # grep L2 ex3ru.out
    # grep Linfty ex3ru.out
    # echo "--------------------------------------------------------------"
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=1" > ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=2" >> ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=3" >> ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=4" >> ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=5" >> ex3r.out
    # parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=3.0 unstructured=False refinement=6" >> ex3r.out
    # grep L2 ex3r.out
    # grep Linfty ex3r.out
else
    echo "Invalid input. Please enter 1 or 2."
    exit 1
fi

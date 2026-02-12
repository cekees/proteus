echo -n "Enter IFEM order to test (1 or 2): "
read num

if [ "$num" -eq 1 ]; then
	echo "Test=9.0 IFEM order 1" > p1.out
	echo "--------------------------------------------------------------"
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=1" > ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=2" >> ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=3" >> ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=4" >> ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=5" >> ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=6" >> ex9.out
	grep L2 ex9.out >> p1.out
	grep Linfty ex9.out >> p1.out
	# echo "--------------------------------------------------------------"
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=1" > ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=2" >> ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=3" >> ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=4" >> ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=5" >> ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=6" >> ex9ru.out
	# grep L2 ex9ru.out
	# grep Linfty ex9ru.out
	# echo "--------------------------------------------------------------"
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=1" > ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=2" >> ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=3" >> ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=4" >> ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=5" >> ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p1_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=6" >> ex9r.out
	# grep L2 ex9r.out
	# grep Linfty ex9r.out
elif [ "$num" -eq 2 ]; then
	echo "Test=9.0 IFEM order 2" > p2.out
	echo "--------------------------------------------------------------"
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=1" > ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=2" >> ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=3" >> ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=4" >> ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=5" >> ex9.out
	parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True refinement=6" >> ex9.out
	grep L2 ex9.out >> p2.out
	grep Linfty ex9.out >> p2.out
	# echo "--------------------------------------------------------------"
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=1" > ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=2" >> ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=3" >> ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=4" >> ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=5" >> ex9ru.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=True skew=0.0 refinement=6" >> ex9ru.out
	# grep L2 ex9ru.out
	# grep Linfty ex9ru.out
	# echo "--------------------------------------------------------------"
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=1" > ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=2" >> ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=3" >> ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=4" >> ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=5" >> ex9r.out
	# parun ladr_ss_2d_p.py ladr_ss_2d_c0p2_n.py -l 5 -v -C "test=9.0 unstructured=False refinement=6" >> ex9r.out
	# grep L2 ex9r.out
	# grep Linfty ex9r.out
else
	echo "Invalid input. Please enter 1 or 2."
	exit 1
fi

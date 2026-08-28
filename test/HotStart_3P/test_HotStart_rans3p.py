"""Tests for 2d flow around a cylinder with a conforming mesh and rans3p"""
from importlib import reload
from proteus.iproteus import *
from proteus import Comm
from proteus import Context
import h5py
import importlib

comm = Comm.get()
Profiling.logLevel = 7
Profiling.verbose = False
import numpy as np


class Test_HotStart_rans3p(object):

    @classmethod
    def setup_class(cls):
        cls._scriptdir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0,cls._scriptdir)
    @classmethod
    def teardown_class(cls):
        sys.path.remove(cls._scriptdir)
        pass

    def setup_method(self, method):
        """Initialize the test problem. """
        self.aux_names = []

    def teardown_method(self, method):
        pass


    def test_hotstart_p1(self):
        self.compare_name = "T01P1_hotstart"
        self.example_setting("T=0.1 vspaceOrder=1 onlySaveFinalSolution=True",h5_filename="solution_p1")
        self.example_setting("T=0.1 vspaceOrder=1 onlySaveFinalSolution=True isHotStart=True", h5_filename="solution_p1", check_result=True, isHotstart=True,hotstart_t=0.1)

    def test_hotstart_p2(self):
        self.compare_name = "T01P2_hotstart"
        self.example_setting("T=0.1 vspaceOrder=2 onlySaveFinalSolution=True",h5_filename="solution_p2")
        self.example_setting("T=0.1 vspaceOrder=2 onlySaveFinalSolution=True isHotStart=True", h5_filename="solution_p2", check_result=True, isHotstart=True,hotstart_t=0.1)


    def example_setting(self, pre_setting, h5_filename, check_result=False, isHotstart=False, hotstart_t=0.0):
        Context.contextOptionsString = pre_setting
        from . import NS_hotstart_so as my_so
        reload(my_so)
        # The p/n modules below (twp_navier_stokes_p.py etc.) import NS_hotstart
        # via a bare "from NS_hotstart import *", not the package-relative
        # "from . import NS_hotstart" above -- these are two separate entries
        # in sys.modules for the same file. reload(my_so) only refreshes the
        # package-relative one, so the bare one (and everything that did
        # "from NS_hotstart import *" from it) stays stale across tests with
        # different context options unless it's reloaded here too.
        if "NS_hotstart" in sys.modules:
            reload(sys.modules["NS_hotstart"])
        # defined in iproteus
        opts.profile = False
        opts.gatherArchive = True
        opts.hotStart = isHotstart
        opts.hotStartTime = hotstart_t
        
        pList=[]
        nList=[]
        sList=[]
        for (pModule,nModule) in my_so.pnList:
            pList.append(
                importlib.import_module(pModule))
            nList.append(
                importlib.import_module(nModule))
            if pList[-1].name == None:
                pList[-1].name = pModule
            reload(pList[-1])  # Serious error
            reload(nList[-1])
        if my_so.sList == []:
            for i in range(len(my_so.pnList)):
                s = default_s
                sList.append(s)
        else:
            sList = my_so.sList
        my_so.name = h5_filename#"_hotstart_"+self.compare_name #save data with different filename
        # NUMERICAL SOLUTION #
        ns = proteus.NumericalSolution.NS_base(my_so,
                                               pList,
                                               nList,
                                               sList,
                                               opts)
        self.aux_names.append(ns.modelList[0].name)
        ns.calculateSolution(my_so.name)
        if check_result:
            # COMPARE VS SAVED FILES #
            actual= h5py.File( my_so.name + '.h5')
            expected_path = 'comparison_files/' + 'comparison_' + self.compare_name + '_u_t2.csv'
            #write comparison file
            #np.array(actual.root.u_t2).tofile(os.path.join(self._scriptdir, expected_path),sep=",")
            # Relative, not absolute-to-10-decimals. decimal=10 demands agreement
            # to 1e-10 on a nonlinear PDE solve, which requires near-identical
            # arithmetic and so is not portable across toolchains: darcy's pip
            # pathway (conda-forge compilers, Intel macOS) drifts ~1e-5 relative
            # -- 845 of 973 elements "mismatched" -- while the same commit on the
            # same host passes under hpc-miniforge and hpc-venv. The measured
            # worst case there is a max relative difference of 1.77e-4 over 86 of
            # 973 elements (max absolute 1.07e-5); rtol=1e-3 sits comfortably
            # above it while still asserting agreement to 0.1%, which is far
            # tighter than any change that would matter physically. Note the
            # maximum only became visible on switching to assert_allclose --
            # assert_almost_equal reports mismatch counts but not the extremes,
            # so sizing a tolerance from its output understates the tail.
            np.testing.assert_allclose(np.fromfile(os.path.join(self._scriptdir, expected_path),sep=","),
                                       np.array(actual['u_t2']).flatten(),
                                       rtol=1.0e-3, atol=1.0e-8)

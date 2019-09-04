import numpy as np
import unittest
import sys
sys.path.insert(0, '../examples')
sys.path.insert(0, '..')
from lorenz63 import Runner
from clv import CLV
class ClvTest(unittest.TestCase):
    def setUp(self):
        self.runner = Runner()
        self.CLV = CLV()

    def test_les(self):
        les, clvs = self.CLV.compute_les_and_clvs()
        les_benchmark = array([0.906, 1.e-8, -14.572])
        err_les = abs((les_benchmark - les)/les_benchmark)
        print("Computed LEs" , les)
        print("Benchmark LEs", les_benchmark)
        print(err_les)
        self.assertTrue(max(err_les[0],err_les[2])<0.2)

    def test_clv_growth(self):
        les, clvs = self.CLV.compute_les_and_clvs()
        tangents = clvs[0]
        primal = self.CLV.stateZero
        nSteps_test = 1000
        d_u = self.CLV.subspace_dim
        runner = self.runner
        parameter = 0.
        les_from_clvs = empty_like(les)
        for i in range(nSteps_test):
            for j in range(d_u):
                tangents[j], sensitivity = runner.tangentSolver(\
                    tangents[j], primal, parameter, 1,\
                    homogeneous=True)
            tangents = tangents.T
            tangents /= linalg.norm(tangents,axis=1)
            print(tangents.T)
            print(clvs[i+1])
            primal, objectiveTrj = runner.primalSolver(primal, \
                    parameter, 1)



if __name__=="__main__":
    unittest.main()

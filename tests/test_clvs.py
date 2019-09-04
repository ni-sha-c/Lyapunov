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
        d_u = self.CLV.subspace_dim
        les_benchmark = les_benchmark[:d_u]
        err_les = abs((les_benchmark - les)/les_benchmark)
        print("Computed LEs" , les)
        print("Benchmark LEs", les_benchmark)
        print(err_les)
        if d_u > 2:
            self.assertTrue(max(err_les[0],err_les[2])<0.2)
        self.assertTrue(err_les[0]<0.2)
    def test_clv_growth(self):
        les, clvs = self.CLV.compute_les_and_clvs()
        nSteps = self.CLV.nSteps_backward
        d_u = self.CLV.subspace_dim
        d = self.runner.state_dim
        self.assertEqual(clvs.shape, (nSteps, d, d_u))
        tangents = clvs[0].T
        tangents_n = clvs[nSteps//2].T
        primal = self.CLV.stateZero
        nSteps_test = 5000
        d_u = self.CLV.subspace_dim
        runner = self.runner
        parameter = 0.
        les_from_clvs = zeros_like(les)
        dt = self.runner.dt
        for i in range(nSteps_test):
            for j in range(d_u):
                tangents[j], sensitivity = runner.tangentSolver(\
                    tangents[j], primal, parameter, 1,\
                    homogeneous=True)
            tangents = tangents.T
            tangent_norms = linalg.norm(tangents,axis=0)
            les_from_clvs += log(abs(tangent_norms))/dt/nSteps_test
            tangents /= tangent_norms
            tangents = tangents.T
            primal, objectiveTrj = runner.primalSolver(primal, \
                    parameter, 1)
        print(les_from_clvs)
        print(les)
        print(self.CLV.clvs[0])


if __name__=="__main__":
    unittest.main()

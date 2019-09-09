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
        clvs = self.CLV.clvs
        primal = self.CLV.stateZero
        d_u = self.CLV.subspace_dim
        runner = self.runner
        parameter = 0.
        les_from_clvs = zeros_like(les)
        dt = self.runner.dt
        for i in range(nSteps):
            tangents = clvs[i].T
            for j in range(d_u):
                tangents[j], sensitivity = runner.tangentSolver(\
                    tangents[j], primal, parameter, 1,\
                    homogeneous=True)
            tangents = tangents.T
            tangent_norms = linalg.norm(tangents,axis=0)
            les_from_clvs += log(abs(tangent_norms))/dt/nSteps
            tangents /= tangent_norms
            primal, objectiveTrj = runner.primalSolver(primal, \
                    parameter, 1)
        les_benchmark = array([0.906, 1.e-8, -14.572])
        les_benchmark = les_benchmark[:d_u]
        err_les = abs((les_benchmark - les_from_clvs)/les_benchmark)
        print("LEs from CLVs" , les_from_clvs)
        print("Benchmark LEs", les_benchmark)
        print(err_les)
        if d_u > 2:
            self.assertTrue(max(err_les[0],err_les[2])<0.2)
        self.assertTrue(err_les[0]<0.2)

    def test_les_adj(self):
        self.CLV.backward_steps_adjoint()
        les = self.CLV.lyap_exps_a
        les_benchmark = array([0.906, 1.e-8, -14.572])
        d_u = self.CLV.subspace_dim
        les_benchmark = les_benchmark[:d_u]
        err_les = abs((les_benchmark - les)/les_benchmark)
        print("Computed LEs from adjoint" , les)
        print("Benchmark LEs", les_benchmark)
        if d_u > 2:
            self.assertTrue(max(err_les[0],err_les[2])<0.2)
        self.assertTrue(err_les[0]<0.2)

    def test_clv_growth_adj(self):
        les, clvs = self.CLV.compute_les_and_clvs_adjoint()
        nSteps = self.CLV.nTrj
        d_u = self.CLV.subspace_dim
        d = self.runner.state_dim
        self.assertEqual(clvs.shape, (nSteps, d, d_u))
        clvs = self.CLV.clvs_a
        primal = self.CLV.stateZero_a
        d_u = self.CLV.subspace_dim
        runner = self.runner
        parameter = 0.
        les_from_clvs = zeros_like(les)
        dt = self.runner.dt
        primalTrj = empty((nSteps,d))
        primalTrj[0] = primal 
        for n in range(1,nSteps):
            primalTrj[n], objectiveTrj = runner.primalSolver(primalTrj[n-1], \
                    parameter, 1)
        for i in range(nSteps-1,-1,-1):
            adjoints = clvs[i].T
            for j in range(d_u):
                adjoints[j], sensitivity = runner.adjointSolver(\
                    adjoints[j], primalTrj[i], parameter, 1,\
                    homogeneous=True)
            adjoints = adjoints.T
            adjoint_norms = linalg.norm(adjoints,axis=0)
            les_from_clvs += log(abs(adjoint_norms))/dt/nSteps
            adjoints /= adjoint_norms
        les_benchmark = array([0.906, 1.e-8, -14.572])
        les_benchmark = les_benchmark[:d_u]
        err_les = abs((les_benchmark - les_from_clvs)/les_benchmark)
        print("LEs from adjoint CLVs" , les_from_clvs)
        print("Benchmark LEs", les_benchmark)
        print(err_les)
        if d_u > 2:
            self.assertTrue(max(err_les[0],err_les[2])<0.2)
        self.assertTrue(err_les[0]<0.2)

if __name__=="__main__":
    unittest.main()

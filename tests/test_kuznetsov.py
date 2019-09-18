import numpy as np
import unittest
import sys
sys.path.insert(0, '../examples')
from kuznetsov_poincare import Runner
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
class KuznetsovTest(unittest.TestCase):
    def setUp(self):
        self.runner = Runner()
    '''
    def test_attractor(self):
        runner = self.runner
        nSteps = 500
        initFields = runner.u_init
        parameter = 0.
        initFields, objectiveTrj = runner.primalSolver(initFields,\
            parameter, nSteps)
        parameters = np.linspace(0., 1.5, 50)
        nSteps = 5000
        objectiveTrend = np.empty_like(parameters)
        for i, parameter in enumerate(parameters):
            initFields, objectiveTrj = runner.primalSolver(initFields,\
                parameter, nSteps)
            objectiveTrend[i] = np.mean(objectiveTrj)
        fig, ax = subplots(1,1,figsize=(8,8))
        ax.plot(parameters,objectiveTrend,lw=3.0)
        ax.set_xlabel("parameter",fontsize=24)
        ax.set_ylabel("time-averaged objective",fontsize=24)
        ax.tick_params(labelsize=24)
    '''
        
   
    def test_tangentSolver(self):
        nSteps = 500
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initPrimal, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, nSteps)
        epss = np.logspace(-10, -3, 10)
        nSteps = 5
        initTangent = np.random.rand(d)
        errs = np.empty_like(epss)
        print("Testing homogeneous tangent")
        finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, nSteps, homogeneous=True)
        for i, eps in enumerate(epss):
            finalPrimal0, objectiveTrj0 = runner.primalSolver(initPrimal - \
                    eps*initTangent, parameter, nSteps)
            finalPrimal, objectiveTrj = runner.primalSolver(initPrimal + \
                    eps*initTangent, parameter, nSteps)

            errs[i] = np.linalg.norm((finalPrimal[:-1] - finalPrimal0[:-1])/(2.0*eps) \
                    - finalTangent[:-1])
        fig, ax = subplots(1,1,figsize=(8,8))
        ax.loglog(epss, errs, "o-", lw=3.0)
        ax.set_xlabel("epsilon used in FD", fontsize=24)
        ax.set_ylabel("err b/w FD and tangent", fontsize=24)
        ax.tick_params(labelsize=24)
        self.assertTrue(np.max(abs(errs))<1.e-1)
        '''
        print("Testing inhomogeneous tangent")
        initTangent = np.zeros_like(initPrimal)
        finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, nSteps)
        for i, eps in enumerate(epss):
            finalPrimal0, objectiveTrj0 = runner.primalSolver(initPrimal,\
                    parameter - eps, nSteps)
            finalPrimal, objectiveTrj = runner.primalSolver(initPrimal,\
                    parameter + eps, nSteps)
            sensitivity_fd = 1./(2*eps)*np.sum(objectiveTrj[:-1] - \
                    objectiveTrj0[:-1])/nSteps
            errs[i] = np.abs(sensitivity_fd - sensitivity)
        
        self.assertTrue(np.max(abs(errs))<1.e-4)
        '''

    def test_adjointSolver(self):
        print("Testing homogeneous adjoint")
        nSteps = 10
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initAdjoint = np.random.rand(d)
        initTangent = np.random.rand(d)
        finalTangent, tangentSensitivity = runner.tangentSolver(\
                initTangent, initPrimal, parameter, nSteps, \
                homogeneous=True)
        finalAdjoint, adjointSensitivity = runner.adjointSolver(\
                initAdjoint, initPrimal, parameter, nSteps, \
                homogeneous=True)
        initDotProduct = np.dot(finalAdjoint, initTangent)
        finalDotProduct = np.dot(initAdjoint, finalTangent)
        print("Tangent-Adjoint dot product at time {0} is {1}".format(\
                0, initDotProduct))
        print("Tangent-Adjoint dot product at time {0} is {1}".format(\
                nSteps, finalDotProduct))
        self.assertTrue(np.abs(initDotProduct - finalDotProduct) < 1.e-10)
        '''
        print("Testing inhomogeneous adjoint")
        initTangent = np.zeros_like(initPrimal)
        initAdjoint = np.zeros_like(initPrimal)
        finalTangent, tangentSensitivity = runner.tangentSolver(\
                initTangent, initPrimal, parameter, nSteps)
        finalAdjoint, adjointSensitivity = runner.adjointSolver(\
                initAdjoint, initPrimal, parameter, nSteps)
        print("Tangent sensitivity is ", tangentSensitivity)
        print("Adjoint sensitivity is ", adjointSensitivity)
'''
if __name__ =="__main__":
    unittest.main()







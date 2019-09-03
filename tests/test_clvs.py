import numpy as np
import unittest
import sys
sys.path.insert(0, '../examples')
from lorenz63 import Runner

class ClvTest(unittest.TestCase):
    def setUp(self):
        self.runner = Runner()
    def test_attractor(self):
        runner = self.runner
        nSteps = 500
        initFields = np.random.rand(3)
        parameter = 0.
        initFields, objectiveTrj = runner.primalSolver(initFields,\
            parameter, nSteps)
        parameters = np.linspace(-10, 50, 50)
        nSteps = 5000
        objectiveTrend = np.empty_like(parameters)
        for i, parameter in enumerate(parameters):
            initFields, objectiveTrj = runner.primalSolver(initFields,\
                parameter, nSteps)
            objectiveTrend[i] = np.mean(objectiveTrj)
        p1, f1 = parameters[1], objectiveTrend[1]
        p2, f2 = parameters[-2], objectiveTrend[-2]
        slope = (f2 - f1)/(p2 - p1)
        intercept = f1 - slope*p1
        objectiveLineFit = slope*parameters + intercept
        self.assertTrue(np.max(abs(objectiveLineFit - objectiveTrend)) \
                <= 1.0)
    
    def test_tangentSolver(self):
        nSteps = 500
        initPrimal = np.random.rand(3)
        parameter = 0.
        runner = self.runner
        initPrimal, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, nSteps)
        epss = np.logspace(-10, -1, 10)
        nSteps = 5
        initTangent = np.random.rand(3)
        errs = np.empty_like(epss)
        print("Testing homogeneous tangent")
        finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, nSteps, homogeneous=True)
        for i, eps in enumerate(epss):
            finalPrimal0, objectiveTrj0 = runner.primalSolver(initPrimal - \
                    eps*initTangent, parameter, nSteps)
            finalPrimal, objectiveTrj = runner.primalSolver(initPrimal + \
                    eps*initTangent, parameter, nSteps)

            errs[i] = np.linalg.norm((finalPrimal - finalPrimal0)/(2.0*eps) \
                    - finalTangent)
        self.assertTrue(np.max(abs(errs))<1.e-4)
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


    def test_adjointSolver(self):
        print("Testing homogeneous adjoint")
        initPrimal = np.random.rand(3)
        nSteps = 10
        parameter = 0.
        runner = self.runner
        initPrimal, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, nSteps)
        initAdjoint = np.random.rand(3)
        initTangent = np.random.rand(3)
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
        print("Testing inhomogeneous adjoint")
        initTangent = np.zeros_like(initPrimal)
        initAdjoint = np.zeros_like(initPrimal)
        finalTangent, tangentSensitivity = runner.tangentSolver(\
                initTangent, initPrimal, parameter, nSteps)
        finalAdjoint, adjointSensitivity = runner.adjointSolver(\
                initAdjoint, initPrimal, parameter, nSteps)
        print("Tangent sensitivity is ", tangentSensitivity)
        print("Adjoint sensitivity is ", adjointSensitivity)

if __name__ =="__main__":
    unittest.main()







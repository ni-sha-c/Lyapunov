import numpy as np
import sys
sys.path.insert(0, '../examples')
from lorenz63 import Runner

runner = Runner()
def test_attractor():
    nSteps = 500
    initFields = rand(3)
    parameter = 0.
    initFields, objectiveTrj = runner.primalSolver(initFields,\
            parameter, nSteps)
    parameters = linspace(-10, 50, 50)
    nSteps = 5000
    objectiveTrend = empty_like(parameters)
    for i, parameter in enumerate(parameters):
        initFields, objectiveTrj = runner.primalSolver(initFields,\
                parameter, nSteps)
        objectiveTrend[i] = np.mean(objectiveTrj)
    p1, f1 = parameters[1], objectiveTrend[1]
    p2, f2 = parameters[-2], objectiveTrend[-2]
    slope = (f2 - f1)/(p2 - p1)
    intercept = f1 - slope*p1
    objectiveLineFit = slope*parameters + intercept
    assert(np.max(abs(objectiveLineFit - objectiveTrend)) <= \
            1.0)
    return
    
def test_tangentSolver():
    nSteps = 500
    initPrimal = rand(3)
    parameter = 0.
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
    assert(np.max(abs(errs))<1.e-4)
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
    
    assert(np.max(abs(errs))<1.e-4)









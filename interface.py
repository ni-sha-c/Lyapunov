import numpy as np
class SerialRunner(Runner):
    def __init__(self, *args, **kwargs):
        return
    
    def runPrimal(self, initFields, primalData, \
            case, args=None):
        parameter, nSteps = primalData
        finalFields, objectiveSeries = Runner.primalSolver(\
                initFields, parameter, nSteps)
        return finalFields, objectiveSeries
    
    def runAdjoint(self, initAdjointFields, primalData, \
            initPrimalFields, case, homogeneous=False, \
            interprocess=None, args=None):
        parameter, nSteps = primalData
        
        return finalFields, dJds

    def runTangent(self, initAdjointFields, primalData, \
            initPrimalFields, case, homogeneous=False, \
            interprocess=None, args=None):
        parameter, nSteps = primalData
        
        return finalFields, dJds


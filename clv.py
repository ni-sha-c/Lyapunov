from numpy import *
import sys
sys.path.insert(0,'examples/')
from lorenz63 import Runner

runner = Runner()

if __name__=="__main__":
    nSpinUp = 500
    parameter = 0.
    initFields = rand(3)
    initFields, objectiveTrj = runner.primalSolver(initFields,\
            parameter, nSpinUp)
    nSteps = 10000
    finalFields, objectiveTrj = runner.primalSolver(initFields,\
            parameter, nSteps)
    d = initFields.shape[0]
    d_u = 2
    tangents_mt1 = rand(d_u, d)
    tangents_mt1, R = np.linalg.qr(tangents_mt1.T)
    tangents = tangents_mt1.T
    primal = finalFields
    lyap_exps = zeros(d_u)
    for i in range(nSteps):
        for j in range(d_u):
            tangents[j], sensitivity = runner.tangentSolver(\
                    tangents[j], primal, parameter, 1,\
                    homogeneous=True)
        tangents, R = np.linalg.qr(tangents.T)
        lyap_exps += log(abs(diag(R)))/nSteps/runner.dt
        tangents = tangents.T
        primal, objectiveTrj = runner.primalSolver(primal, \
                parameter, 1)






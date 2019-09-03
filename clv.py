from numpy import *
import sys
sys.path.insert(0,'examples/')
from lorenz63 import Runner

runner = Runner()

if __name__=="__main__":
    nSpinUp = 500
    parameter = 0.
    stateInit = rand(3)
    stateInit, objectiveTrj = runner.primalSolver(stateInit,\
            parameter, nSpinUp)
    nSteps = 10000
    nSteps_forward = nSteps//2
    nSteps_backward = nSteps//2
    stateZero, objectiveTrj = runner.primalSolver(stateInit,\
            parameter, nSteps_forward)
    stateT, objectiveTrj = runner.primalSolver(stateZero, \
            parameter, nSteps_backward)
    d = stateInit.shape[0]
    d_u = 3
    tangents_mt1 = rand(d_u, d)
    tangents_mt1, R = np.linalg.qr(tangents_mt1.T)
    tangents = tangents_mt1.T
    lyap_exps = zeros(d_u)
    primal = copy(stateInit)
    RTrj = empty((nSteps_backward,d_u,d_u))
    QTrj = empty((nSteps_backward,d,d_u))
    clvs = empty((nSteps_backward,d,d_u))
    coeffsTrj = empty((nSteps_backward,d_u,d_u))
    for i in range(nSteps_forward):
        for j in range(d_u):
            tangents[j], sensitivity = runner.tangentSolver(\
                    tangents[j], primal, parameter, 1,\
                    homogeneous=True)
        tangents, R = np.linalg.qr(tangents.T)
        lyap_exps += log(abs(diag(R)))/nSteps/runner.dt
        tangents = tangents.T
        primal, objectiveTrj = runner.primalSolver(primal, \
                parameter, 1)
    for i in range(nSteps_backward):
        RTrj[i] = R
        QTrj[i] = tangents
        for j in range(d_u):
            tangents[j], sensitivity = runner.tangentSolver(\
                    tangents[j], primal, parameter, 1,\
                    homogeneous=True)
        tangents, R = np.linalg.qr(tangents.T)
        lyap_exps += log(abs(diag(R)))/nSteps/runner.dt
        tangents = tangents.T
        primal, objectiveTrj = runner.primalSolver(primal, \
                parameter, 1) 
    coeffsTrj[-1] = eye(d_u,d_u)
    clvs[-1] = QTrj[-1]
    for i in range(nSteps_backward-1,0,-1):
        coeffsTrj[i-1] = linalg.solve(RTrj[i], coeffsTrj[i]) 
        coeffsTrj[i-1] /= norm(coeffsTrj[i-1],axis=0)
        clvs[i-1] = dot(QTrj[i-1], coeffsTrj[i-1])
        clvs[i-1] /= norm(clvs[i-1],axis=0)





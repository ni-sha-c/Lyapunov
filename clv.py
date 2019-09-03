from numpy import *
import sys
sys.path.insert(0,'examples/')
from lorenz63 import Runner

class CLV():
    def init(self):
        self.runner = Runner()
        self.subspace_dim = 3

    def setup(self):
        nSpinUp = 500
        parameter = 0.
        runner = self.runner
        d = runner.state_dim
        stateInit = rand(d)
        stateInit, objectiveTrj = runner.primalSolver(stateInit,\
            parameter, nSpinUp)
        d_u = self.subspace_dim
        self.nSteps = 10000
        self.nSteps_forward = self.nSteps//2
        self.nSteps_backward = self.nSteps//2
        nSteps_backward = self.nSteps_backward
        tangents_mt1 = rand(d_u, d)
        tangents_mt1, R = np.linalg.qr(tangents_mt1.T)
        self.tangents = tangents_mt1.T
        self.lyap_exps = zeros(d_u)
        self.primal = stateInit
        self.RTrj = empty((nSteps_backward,d_u,d_u))
        self.QTrj = empty((nSteps_backward,d,d_u))
        self.clvs = empty((nSteps_backward,d,d_u))
        self.coeffsTrj = empty((nSteps_backward,d_u,d_u))
        return

    def get_most_expanding_directions(self):
        self.setup()
        nSteps = self.nSteps
        nSteps_forward = self.nSteps_forward
        nSteps_backward = self.nSteps_backward
        primal = self.primal
        runner = self.runner
        tangents = self.tangents
        lyap_exps = self.lyap_exps

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
        self.stateZero = copy(primal)
        return

    def forward_steps(self):
        self.get_most_expanding_directions()
        RTrj = self.RTrj
        QTrj = self.QTrj
        lyap_exps = self.lyap_exps
        runner = self.runner
        tangents = self.tangents
        primal = self.primal
        for i in range(self.nSteps_backward):
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

    def backward_steps(self):
        self.forward_steps()
        self.coeffsTrj[-1] = eye(d_u,d_u)
        self.clvs[-1] = self.QTrj[-1]
        for i in range(self.nSteps_backward-1,0,-1):
            self.coeffsTrj[i-1] = linalg.solve(self.RTrj[i], \
                    self.coeffsTrj[i]) 
            self.coeffsTrj[i-1] /= norm(self.coeffsTrj[i-1],axis=0)
            self.clvs[i-1] = dot(self.QTrj[i-1], self.coeffsTrj[i-1])
            self.clvs[i-1] /= norm(self.clvs[i-1],axis=0)

    def compute_les_and_clvs(self):
        self.backward_steps()
        return self.lyap_exps



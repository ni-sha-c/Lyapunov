from numpy import *
import sys
sys.path.insert(0,'examples/')
from lorenz63 import Runner
class CLV():
    def __init__(self,d_u=3,nTrj=500):
        self.runner = Runner()
        self.subspace_dim = d_u
        self.nTrj = nTrj


    def setup_tangents(self):
        nSpinUp = 500
        parameter = 0.
        runner = self.runner
        d = runner.state_dim
        stateInit = random.rand(d)
        stateInit, objectiveTrj = runner.primalSolver(stateInit,\
            parameter, nSpinUp)
        d_u = self.subspace_dim
        self.nSteps = 10000

        tangents_mt1 = random.rand(d_u, d)
        tangents_mt1, R = linalg.qr(tangents_mt1.T)
        self.R = R
        self.tangents = tangents_mt1.T
        self.lyap_exps = zeros(d_u)
        self.primal = stateInit
        self.nSteps_backward = self.nTrj
        nSteps_backward = self.nSteps_backward
        self.nSteps_forward = self.nSteps - nSteps_backward

        self.RTrj = empty((nSteps_backward,d_u,d_u))
        self.QTrj = empty((nSteps_backward,d,d_u))
        self.clvs = empty((nSteps_backward,d,d_u))
        self.coeffsTrj = empty((nSteps_backward,d_u,d_u))
        return

    def setup_adjoints(self):
        runner = self.runner
        d = runner.state_dim
        d_u = self.subspace_dim
        stateInit = random.rand(d)
        parameter = 0.
        nSpinUp = 500
        self.nSteps = 10000
        stateInit, objectiveTrj = runner.primalSolver(stateInit,\
            parameter, nSpinUp)
        self.stateZero = stateInit
        adjoints_mt1 = random.rand(d_u, d)
        adjoints_mt1, R = linalg.qr(adjoints_mt1.T)
        self.R_a = R
        self.adjoints = adjoints_mt1.T
        self.lyap_exps_a = zeros(d_u)
        nTrj = self.nTrj
        self.nSteps_backward_a = self.nSteps - nTrj
        self.RTrj_a = empty((nTrj,d_u,d_u))
        self.QTrj_a = empty((nTrj,d,d_u))
        self.clvs_a = empty((nTrj,d,d_u))
        self.coeffsTrj_a = empty((nTrj,d_u,d_u))
        return


    def get_most_expanding_directions(self):
        self.setup_tangents()
        nSteps = self.nSteps
        nSteps_forward = self.nSteps_forward
        nSteps_backward = self.nSteps_backward
        primal = self.primal
        runner = self.runner
        tangents = self.tangents
        lyap_exps = self.lyap_exps
        R = self.R
        d_u = self.subspace_dim
        parameter = 0.
        for i in range(nSteps_forward):
            for j in range(d_u):
                tangents[j], sensitivity = runner.tangentSolver(\
                    tangents[j], primal, parameter, 1,\
                    homogeneous=True)
            tangents, R = linalg.qr(tangents.T)
            lyap_exps += log(abs(diag(R)))/nSteps/runner.dt
            tangents = tangents.T
            primal, objectiveTrj = runner.primalSolver(primal, \
                parameter, 1)
        self.stateZero = copy(primal)
        self.primal = primal
        self.tangents = tangents
        self.R = R
        self.lyap_exps = lyap_exps
        
    

    def forward_steps(self):
        self.get_most_expanding_directions()
        RTrj = self.RTrj
        QTrj = self.QTrj
        lyap_exps = self.lyap_exps
        runner = self.runner
        tangents = self.tangents
        primal = self.primal
        primalTrj = copy(self.primal)
        R = self.R
        parameter = 0.
        d_u = self.subspace_dim
        nSteps = self.nSteps
        for i in range(self.nSteps_backward):
            RTrj[i] = R
            QTrj[i] = tangents.T
            for j in range(d_u):
                tangents[j], sensitivity = runner.tangentSolver(\
                    tangents[j], primal, parameter, 1,\
                    homogeneous=True)
            tangents, R = linalg.qr(tangents.T)
            lyap_exps += log(abs(diag(R)))/nSteps/runner.dt
            tangents = tangents.T
            primal, objectiveTrj = runner.primalSolver(primal, \
                parameter, 1) 
             
        self.primal = primal
        self.tangents = tangents
        self.R = R
        self.QTrj = QTrj
        self.RTrj = RTrj
        self.lyap_exps = lyap_exps


    def backward_steps(self):
        self.forward_steps()
        d_u = self.subspace_dim
        self.coeffsTrj[-1] = eye(d_u,d_u)
        self.clvs[-1] = self.QTrj[-1]
        lyap_exps = exp(self.lyap_exps*self.runner.dt)
        for i in range(self.nSteps_backward-1,0,-1):
            self.coeffsTrj[i-1] = linalg.solve(self.RTrj[i], \
                    self.coeffsTrj[i]) 
            self.coeffsTrj[i-1] *= lyap_exps
            self.clvs[i-1] = dot(self.QTrj[i-1], self.coeffsTrj[i-1])
            self.clvs[i-1] /= linalg.norm(self.clvs[i-1],axis=0)
   

    def backward_steps_adjoint(self):
        self.setup_adjoints()
        runner = self.runner 
        d = runner.state_dim
        d_u = self.subspace_dim
        nSteps = self.nSteps 
        adjointSolver = runner.adjointSolver
        primalTrj = empty((nSteps, d))
        primalTrj[0] = self.stateZero
        parameter = 0.
        dt = self.runner.dt
        for n in range(1,nSteps):
            primalTrj[n], objectiveTrj = runner.primalSolver(primalTrj[n-1], \
                parameter, 1)
        adjoints = random.rand(d_u, d)
        adjoints, R = linalg.qr(adjoints.T)
        adjoints = adjoints.T
        lyap_exps_a = self.lyap_exps_a
        for n in range(nSteps-1,-1,-1):
            for j in range(d_u):
                adjoints[j], sensitivity = adjointSolver(adjoints[j], primalTrj[n], \
                        parameter, 1, True)
            adjoints, R = linalg.qr(adjoints.T)
            adjoints = adjoints.T
            lyap_exps_a += log(abs(diag(R)))/dt/nSteps 
        return lyap_exps_a





    def compute_les_and_clvs(self):
        self.backward_steps()
        return self.lyap_exps, self.clvs



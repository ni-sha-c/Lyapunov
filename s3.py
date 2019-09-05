from numpy import *
import sys
sys.path.insert(0,'examples/')
from lorenz63 import Runner

class S3():
    def __init__(self):
        self.runner = Runner()
        self.subspace_dim = 3

    def setup(self):
        nSpinUp = 500
        parameter = 0.
        runner = self.runner
        d = runner.state_dim
        stateInit = random.rand(d)
        stateInit, objectiveTrj = runner.primalSolver(stateInit,\
            parameter, nSpinUp)
        d_u = self.subspace_dim
        self.nSteps = 100
        self.nSteps_forward = self.nSteps//2
        self.nSteps_backward = self.nSteps//2
        nSteps_backward = self.nSteps_backward
        tangents_mt1 = random.rand(d_u, d)
        tangents_mt1, R = linalg.qr(tangents_mt1.T)
        self.R = R
        self.tangents = tangents_mt1.T
        self.lyap_exps = zeros(d_u)
        self.primal = stateInit
        self.RTrj = empty((nSteps_backward,d_u,d_u))
        self.QTrj = empty((nSteps_backward,d,d_u))
        self.clvs = empty((nSteps_backward,d,d_u))
        self.coeffsTrj = empty((nSteps_backward,d_u,d_u))
        return

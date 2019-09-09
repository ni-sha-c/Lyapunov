from numpy import *
import sys
sys.path.insert(0,'examples/')
from lorenz63 import Runner
from clv import CLV
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
        self.CLV = CLV(d_u,nSteps)
        self.les, self.clvs = self.CLV.compute_les_and_clvs()
        self.primal = self.CLV.stateZero



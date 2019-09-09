from numpy import *
import sys
sys.path.insert(0,'examples/')
from lorenz63 import Runner
from clv import CLV
class S3():
    def __init__(self):
        self.runner = Runner()
        self.subspace_dim = 3
        self.nSteps = 1000

    def setup(self):
        nSpinUp = 500
        parameter = 0.
        runner = self.runner
        d = runner.state_dim
        d_u = self.subspace_dim
        nSteps = self.nSteps
        self.CLV = CLV(d_u, nSteps)
        self.les_t, self.clvs_t = self.CLV.compute_les_and_clvs()
        self.primalInit = self.CLV.stateZero
        self.les_a, self.clvs_a = self.CLV.compute_les_and_clvs_adjoint(self.primalInit)
        dt = self.runner.dt
        self.les_a *= dt
        self.les_t *= dt
        d_ua = (self.les_a > 0).sum()
        d_ut = (self.les_t > 0).sum()
        self.subspace_dim = max(d_ua, d_ut)
        self.clvs_t = self.clvs_t[:,:,:self.subspace_dim]
        self.clvs_a = self.clvs_a[:,:,:self.subspace_dim]

    def main(self):
        self.setup()
        nSteps = self.nSteps
        primal = copy(self.primalInit) 
        runner = self.runner
        parameter = 0.
        d_u = self.subspace_dim
        d = self.runner.state_dim
        angles = zeros((d_u, d_u))
        stable_contrib = 0.
        unstable_contrib = 0.
        zeta_s = zeros(d)
        for n in range(nSteps):
            X = runner.source(primal, parameter)
            V = self.clvs_t[n].T
            W = self.clvs_a[n].T
            cos_angles = zeros((d_u, d_u))
            for i in range(d_u):
                for j in range(d_u):
                    cos_angles[i, j] = dot(V[j], W[i])
            X_dot_W = dot(W, X)  
            X_along_V = linalg.solve(cos_angles, X_dot_W)
            X_u = dot(V.T, X_along_V)
            X_s = X - X_u
            zeta_s, sensitivity = runner.tangentSolver(zeta_s, primal,\
                    parameter, 1, True)

            zeta_s += X_s
            zetas_dot_W = dot(W, zeta_s)
            zetas_along_V = linalg.solve(cos_angles, zetas_dot_W)
            zetas_u = dot(V.T, zetas_along_V)
            zeta_s = zeta_s - zetas_u


            for i in range(d_u):
                zeta_s -= dot(zeta_s, W[i])*W[i]
            primal, objectiveTrj = runner.primalSolver(primal, parameter, 1)
            D_obj = runner.gradientObjective(primal, parameter)
            stable_contrib += dot(zeta_s, D_obj)/nSteps 
        stop 
             



            






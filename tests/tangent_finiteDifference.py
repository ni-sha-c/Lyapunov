import numpy as np
import unittest
import sys
sys.path.insert(0, '../examples')
from kuznetsov_poincare import Runner
from matplotlib.pyplot import *
from matplotlib import animation, rc
class KuznetsovTest(unittest.TestCase):
    def setUp(self):
        self.runner = Runner()
    def collate_nearby_points(self):
        nSteps = 2000
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initPrimal0, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, nSteps)
        initTangent0 = np.random.rand(d)
        initTangent0 /= np.linalg.norm(initTangent0)
        initTangent = np.copy(initTangent0)
        initPrimal = np.copy(initPrimal0)
        clv_trj_orig = np.empty((nSteps, d))
        primal_orig = np.empty((nSteps,d))
        for n in range(nSteps):
            primal_orig[n] = initPrimal
            clv_trj_orig[n] = initTangent
            finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, 1, homogeneous=True)
            initPrimal, objectiveTrj = runner.primalSolver(initPrimal, \
                    parameter, 1)
            initTangent = finalTangent/np.linalg.norm(finalTangent)
        initPrimal = initPrimal + 1.e-8*initTangent
        primal_pert = np.empty((nSteps, d))
        clv_trj_pert = np.empty((nSteps, d))
        for n in range(nSteps):
            primal_pert[n] = initPrimal
            clv_trj_orig[n] = initTangent
            finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, 1, homogeneous=True)
            initPrimal, objectiveTrj = runner.primalSolver(initPrimal, \
                    parameter, 1)
            initTangent = finalTangent/np.linalg.norm(finalTangent)
        distances = np.empty((nSteps, nSteps))
        distances_along_clvs = np.empty((nSteps, nSteps))
        closest_times = np.empty(nSteps,dtype=int)
        closest_times_as_per_clvs = np.empty(nSteps,dtype=int)
        for i in range(nSteps):
            distances[i] = np.linalg.norm(primal_pert[i] - primal_orig,\
                    axis=1)
            distances_along_clvs[i] = np.diag(np.abs(np.matmul(\
                    primal_pert[i] - primal_orig, clv_trj_orig.T)))
            closest_times[i] = np.argmin(distances[i])
            closest_times_as_per_clvs[i] = np.argmin(distances_along_clvs[i])
        fig, ax = subplots(1,1,figsize=(8,8))
        ax.set_xlim([-3.0,3.0])
        ax.set_ylim([-3.5,2.0])
        nPlot = 10
        original_re_part, original_im_part = runner.stereographic_projection(\
                primal_orig[closest_times[:nPlot]].T)
        perturbed_re_part, perturbed_im_part = runner.stereographic_projection(\
                primal_pert[:nPlot].T)


        alpha = 0.5
        original, = ax.plot(original_re_part, original_im_part, '.',ms=20, \
                    color="k")
        perturbed, = ax.plot(perturbed_re_part, perturbed_im_part, '.', ms=20, \
                    color="b")
        stop      



    def plot_two_trajectories_history(self):
        nSteps = 1000
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initPrimal0, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, nSteps)
        epss = np.logspace(-10, -3, 10)
        initTangent0 = np.random.rand(d)
        initTangent0 /= np.linalg.norm(initTangent0)
        initTangent = np.copy(initTangent0)
        initPrimal = np.copy(initPrimal0)
        errs = np.empty_like(epss)
        dPrimal_dEpsilon = np.empty((len(epss), d))
        for n in range(nSteps):
            finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, 1, homogeneous=True)
            initPrimal, objectiveTrj = runner.primalSolver(initPrimal, \
                    parameter, 1)
            initTangent = finalTangent/np.linalg.norm(finalTangent)
        nTrack = 5000
        tangent = np.copy(initTangent)
        primal_original = np.copy(initPrimal)
        primal_perturbed = primal_original + epss[3]*initTangent 
        original_trj = np.empty((nTrack, d))
        perturbed_trj = np.empty((nTrack, d))
        clv_trj = np.empty((nTrack, d))
        for n in range(nTrack):
            primal_perturbed, obj = runner.primalSolver(primal_perturbed,\
                    parameter,1)
            perturbed_trj[n] = primal_perturbed
            tangent, sensitivity = runner.tangentSolver(tangent, \
                    primal_original, parameter, 1, homogeneous=True)
            primal_original, obj = runner.primalSolver(primal_original,\
                    parameter,1)
            original_trj[n] = primal_original
            tangent /= np.linalg.norm(tangent)
            clv_trj[n] = tangent


        original_re_part, original_im_part = runner.stereographic_projection(\
                original_trj.T)
        
        perturbed_re_part, perturbed_im_part = runner.stereographic_projection(\
                perturbed_trj.T)

        clv_re_part, clv_im_part = runner.convert_tangent_euclidean_to_stereo(original_trj.T, clv_trj.T)
        norm_clv = clv_re_part*clv_re_part + clv_im_part*clv_im_part
        norm_clv = np.sqrt(norm_clv)
        clv_re_part /= norm_clv
        clv_im_part /= norm_clv

        fig, ax = subplots(1,1,figsize=(8,8))
        ax.set_xlim([-3.0,3.0])
        ax.set_ylim([-3.5,2.0])
        
        alpha = 0.5
        #ax = self.plot_attractor(ax)
        original, = ax.plot(original_re_part, original_im_part, '.',ms=20, \
                    color="k")
        perturbed, = ax.plot(perturbed_re_part, perturbed_im_part, '.', ms=20, \
                    color="b")


    def plot_two_trajectories_animation(self):
        nSteps = 1000
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initPrimal0, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, nSteps)
        epss = np.logspace(-10, -3, 10)
        initTangent0 = np.random.rand(d)
        initTangent0 /= np.linalg.norm(initTangent0)
        initTangent = np.copy(initTangent0)
        initPrimal = np.copy(initPrimal0)
        errs = np.empty_like(epss)
        dPrimal_dEpsilon = np.empty((len(epss), d))
        for n in range(nSteps):
            finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, 1, homogeneous=True)
            initPrimal, objectiveTrj = runner.primalSolver(initPrimal, \
                    parameter, 1)
            initTangent = finalTangent/np.linalg.norm(finalTangent)
        nTrack = 100
        tangent = np.copy(initTangent)
        primal_original = np.copy(initPrimal)
        primal_perturbed = primal_original + epss[3]*initTangent 
        original_trj = np.empty((nTrack, d))
        perturbed_trj = np.empty((nTrack, d))
        clv_trj = np.empty((nTrack, d))
        for n in range(nTrack):
            primal_perturbed, obj = runner.primalSolver(primal_perturbed,\
                    parameter,1)
            perturbed_trj[n] = primal_perturbed
            tangent, sensitivity = runner.tangentSolver(tangent, \
                    primal_original, parameter, 1, homogeneous=True)
            primal_original, obj = runner.primalSolver(primal_original,\
                    parameter,1)
            original_trj[n] = primal_original
            tangent /= np.linalg.norm(tangent)
            clv_trj[n] = tangent


        original_re_part, original_im_part = runner.stereographic_projection(\
                original_trj.T)
        
        perturbed_re_part, perturbed_im_part = runner.stereographic_projection(\
                perturbed_trj.T)

        clv_re_part, clv_im_part = runner.convert_tangent_euclidean_to_stereo(original_trj.T, clv_trj.T)
        norm_clv = clv_re_part*clv_re_part + clv_im_part*clv_im_part
        norm_clv = np.sqrt(norm_clv)
        clv_re_part /= norm_clv
        clv_im_part /= norm_clv

        fig, ax = subplots(1,1,figsize=(8,8))
        ax.set_xlim([-3.0,3.0])
        ax.set_ylim([-3.5,2.0])
        
        alpha = 0.5
        ax = self.plot_attractor(ax)
        original, = ax.plot(original_re_part[0], original_im_part[0], marker=".", linestyle=None, ms=30, \
                    color="k")
        original_clv, = ax.plot([original_re_part[0]-alpha*clv_re_part[0],\
                original_re_part[0]+alpha*clv_re_part[0]],\
                [original_im_part[0]-alpha*clv_im_part[0],\
                original_im_part[0]+alpha*clv_im_part[0]],'r-', lw=3)
        perturbed, = ax.plot(perturbed_re_part[0], perturbed_im_part[0], marker=".", linestyle=None, ms=30, \
                    color="b")


        def update(n):
            original.set_data(original_re_part[n], original_im_part[n])
            original_clv.set_data([original_re_part[n]-alpha*clv_re_part[n],\
                    original_re_part[n]+alpha*clv_re_part[n]],\
                    [original_im_part[n]-alpha*clv_im_part[n],\
                    original_im_part[n]+alpha*clv_im_part[n]])
            perturbed.set_data(perturbed_re_part[n], perturbed_im_part[n])
        anim = animation.FuncAnimation(fig, update, frames=100, repeat=True)
        return anim
    def plot_attractor(self, ax=None):
        nSteps = 10000
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initPrimal0, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, 500)
        if ax==None:
            fig, ax = subplots(1,1,figsize=(8,8))
        initPrimal = np.copy(initPrimal0)
        for n in range(nSteps):
            initPrimal, objectiveTrj = runner.primalSolver(initPrimal, \
                    parameter, 1)
            re_part, im_part = runner.stereographic_projection(initPrimal)
            ax.plot(re_part, im_part, color='gray',\
                    marker='.', ms=5)
        return ax

    '''
    def test_CLV_of_unstable_manifold(self):
        nSteps = 2000
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initPrimal0, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, nSteps)
        epss = np.logspace(-10, -3, 10)
        initTangent0 = np.random.rand(d)
        initTangent0 /= norm(initTangent0)
        initTangent = np.copy(initTangent0)
        initPrimal = np.copy(initPrimal0)
        errs = np.empty_like(epss)
        dPrimal_dEpsilon = np.empty((len(epss), d))
        for n in range(nSteps):
            finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, 1, homogeneous=True)
            initPrimal, objectiveTrj = runner.primalSolver(initPrimal, \
                    parameter, 1)
            initTangent = finalTangent/norm(finalTangent)
            
        finalTangent /= norm(finalTangent)
        nSteps_per_checkpoint = 10
        nCheckpoints = nSteps//nSteps_per_checkpoint
        tangent = initTangent0
        for i, eps in enumerate(epss):
            initPrimal = np.copy(initPrimal0)
            for n in range(nCheckpoints):
                finalPrimal0, objectiveTrj0 = runner.primalSolver(initPrimal,\
                parameter, nSteps_per_checkpoint)
                finalPrimal, objectiveTrj = runner.primalSolver(initPrimal + \
                    eps*tangent, parameter, nSteps_per_checkpoint)
                tangent = (finalPrimal - finalPrimal0)/norm(\
                        finalPrimal - finalPrimal0)
                initPrimal = finalPrimal0

            dPrimal_dEpsilon[i] = tangent
            errs[i] = abs(1.0 - abs(np.dot(dPrimal_dEpsilon[i],\
                    finalTangent)))
        stop
        fig, ax = subplots(1,1,figsize=(8,8))
        ax.loglog(epss, errs, "o-", lw=3.0)
        ax.set_xlabel("epsilon used in FD", fontsize=24)
        ax.set_ylabel("err b/w FD and tangent", fontsize=24)
        ax.tick_params(labelsize=24)
        self.assertTrue(np.max(abs(errs))<1.e-1)

    
    def test_CLV_tangent_finiteDifference(self):
        nSteps = 2000
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initPrimal0, objectiveTrj = runner.primalSolver(initPrimal,\
                parameter, nSteps)
        epss = np.logspace(-10, -3, 10)
        initTangent0 = np.random.rand(d)
        initTangent0 /= norm(initTangent0)
        initTangent = np.copy(initTangent0)
        initPrimal = np.copy(initPrimal0)
        errs = np.empty_like(epss)
        dPrimal_dEpsilon = np.empty((len(epss), d))
        for n in range(nSteps):
            finalTangent, sensitivity = runner.tangentSolver(initTangent, \
                initPrimal, parameter, 1, homogeneous=True)
            initPrimal, objectiveTrj = runner.primalSolver(initPrimal, \
                    parameter, 1)
            initTangent = finalTangent/norm(finalTangent)
            
        finalTangent /= norm(finalTangent)
        nSteps_per_checkpoint = 10
        nCheckpoints = nSteps//nSteps_per_checkpoint
        tangent = initTangent0
        for i, eps in enumerate(epss):
            initPrimal = np.copy(initPrimal0)
            for n in range(nCheckpoints):
                finalPrimal0, objectiveTrj0 = runner.primalSolver(initPrimal,\
                parameter, nSteps_per_checkpoint)
                finalPrimal, objectiveTrj = runner.primalSolver(initPrimal + \
                    eps*tangent, parameter, nSteps_per_checkpoint)
                tangent = (finalPrimal - finalPrimal0)/norm(\
                        finalPrimal - finalPrimal0)
                initPrimal = finalPrimal0

            dPrimal_dEpsilon[i] = tangent
            errs[i] = abs(1.0 - abs(np.dot(dPrimal_dEpsilon[i],\
                    finalTangent)))
        stop
        fig, ax = subplots(1,1,figsize=(8,8))
        ax.loglog(epss, errs, "o-", lw=3.0)
        ax.set_xlabel("epsilon used in FD", fontsize=24)
        ax.set_ylabel("err b/w FD and tangent", fontsize=24)
        ax.tick_params(labelsize=24)
        self.assertTrue(np.max(abs(errs))<1.e-1)
    
    def test_adjointSolver(self):
        print("Testing homogeneous adjoint")
        nSteps = 10
        parameter = 1.
        runner = self.runner
        initPrimal = runner.u_init
        d = runner.state_dim
        initAdjoint = np.random.rand(d)
        initTangent = np.random.rand(d)
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
    '''
if __name__ =="__main__":
    kuz = KuznetsovTest()
    kuz.setUp()
    kuz.collate_nearby_points()
    #anim = kuz.plot_two_trajectories_history()
    #unittest.main()







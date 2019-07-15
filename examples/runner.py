import numpy as np
class Runner(object):
    def __init__(self, *args, **kwargs):
        self.solverName = "Lorenz '63"
        self.sigma = 10.
        self.beta = 8./3.
        self.rho = 28.0
        self.dt = 1.e-2


    def primalSolver(self, initFields, parameter,\
            nSteps):
        x, y, z = initFields
        sigma = self.sigma
        beta = self.beta
        rho = self.rho
        dt = self.dt
        objectiveSeries = np.zeros(nSteps)
        for i in range(nSteps):
            dx_dt = sigma*(y - x)
            dy_dt = x*(parameter + rho - z) - y
            dz_dt = x*y - beta*z
            
            x += dt*dx_dt
            y += dt*dy_dt
            z += dt*dz_dt
            finalFields = np.array([x,y,z])
            objectiveSeries[i] = self.objective(finalFields, parameter)
    
        return finalFields, objectiveSeries

    def objective(self, fields, parameter):
        return fields[-1]

    def source(self, fields, parameter):
        sourceTerms = np.zeros_like(fields)
        sourceTerms[1] = self.dt*fields[0]
        return sourceTerms
        
    def gradientObjective(self, fields, parameter, nSteps):
        dJ = np.zeros_like(fields)
        dJ[-1] = 1.0/nSteps
        return dJ

    def tangentSolver(self, initFields, initPrimalFields, \
            parameter, nSteps, homogeneous=False):
        primalTrj = np.empty(shape=(nSteps, initFields.shape[0]))
        objectiveTrj = np.empty(nSteps)

        primalTrj[0] = initPrimalFields
        objectiveTrj[0] = self.objective(primalTrj[0],parameter)
        for i in range(1, nSteps):
            primalTrj[i], objectiveTrj[i] = self.primalSolver(\
                    primalTrj[i-1], parameter, 1)
        xt, yt, zt = initFields
        sensitivity = 0.
        for i in range(1, nSteps):
            x, y, z = primalTrj[i-1]
            dxt_dt = sigma*(yt - xt) 
            dyt_dt = (parameter + rho - z)*xt - zt*x - yt 
            dzt_dt = x*yt + y*xt - beta*zt
            
            xt += dt*dxt_dt 
            yt += dt*dyt_dt
            zt += dt*dzt_dt
            
            finalFields = np.array([xt, yt, zt])
            if(homogeneous==False):
                finalFields += self.source(primalTrj[i-1],\
                        parameter)
            sensitivity += np.dot(finalFields, \
                    self.gradientObjective(primalTrj[i], parameter, \
                    nSteps))
        return finalFields, sensitivity
            

    def adjointSolver(self, initFields, initPrimalFields, \
            parameter, nSteps, homogeneous=False):
        rho = self.rho
        beta = self.beta
        sigma = self.sigma
        primalTrj = np.empty(shape=(nSteps, initFields.shape[0]))
        objectiveTrj = np.empty(nSteps)

        primalTrj[0] = initAdjointFields
        objectiveTrj[0] = self.objective(primalTrj[0],parameter)
        for i in range(1, nSteps):
            primalTrj[i], objectiveTrj[i] = self.primalSolver(\
                    primalTrj[i-1], parameter, 1)
        xa, ya, za = initFields
        sensitivity = 0.
        for i in range(nSteps, 0, -1):
            x, y, z = primalTrj[i-1]
            dxa_dt = -sigma*xa + (parameter + rho - z)*ya + \
                    y*za 
            dya_dt = sigma*xa - ya + x*za 
            dza_dt = -x*ya - beta*za 
            
            xa += dt*dxa_dt 
            ya += dt*dya_dt
            za += dt*dza_dt
           
            finalFields = np.array([xa, ya, za])
            if(homogeneous==False):
                finalFields += gradientObjective(primalTrj[i-1])
            sensitivity += np.dot(finalFields, self.source(\
                    primalTrj[i-1], parameter))
        return finalFields, sensitivity
            




            

import numpy as np
class Runner(object):
    def __init__(self, *args, **kwargs):
        self.solverName = "Lorenz '63"
        self.sigma = 10.
        self.beta = 8./3.
        self.dt = 1.e-2


    def primalSolver(self, initFields, parameter,\
            nSteps):
        x, y, z = initFields
        sigma = self.sigma
        beta = self.beta
        dt = self.dt
        objectiveSeries = np.zeros(nSteps)
        for i in range(nSteps):
            dx_dt = sigma*(y - x)
            dy_dt = x*(parameter - z) - y
            dz_dt = x*y - beta*z
            
            x += dt*dx_dt
            y += dt*dy_dt
            z += dt*dz_dt
            finalFields = np.array([x,y,z])
            objectiveSeries[i] = self.objective(finalFields, parameter)
    
        return finalField, objectiveSeries

    def objective(self, fields, parameter):
        return fields[-1]

    def adjointSolver(self, initFields, initPrimalFields, \
            parameter, nSteps, homogeneous=False):
        primalTrj = np.empty(shape=(nSteps, initFields.shape[0]))
        objectiveTrj = np.empty(nSteps)

        primalTrj[0] = initFields
        objectiveTrj[0] = self.objective(primalTrj[0],parameter)
        for i in range(1, nSteps):
            primalTrj[i], objectiveTrj[i] = self.primalSolver(\
                    primalTrj[i-1], parameter, 1)
        xa, ya, za = initFields
        for i in range(nSteps, 0, -1):
            x, y, z = primalTrj[i-1]
            dxa_dt = -sigma*xa + (parameter - z)*ya + \
                    y*za 
            dya_dt = sigma*xa - ya + x*za 
            dza_dt = -x*ya - beta*za 
            
            xa += dt*dxa_dt 
            ya += dt*dya_dt
            za += dt*dza_dt
            
            if(homogeneous):
                dJ = gradientObjective(primalTrj[i-1])
                xa += (1/nSteps)*dJ[0]
                ya += (1/nSteps)*dJ[1]
                za += (1/nSteps)*dJ[2]

            




            

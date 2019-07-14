from numpy import *

class Shadowing(SerialRunner):
    def __init__(self, *args, **kwargs):
        super(Shadowing, self).__init__(*args, **kwargs)

    def solve(self, initFields, parameter, nSteps, run_id):
        case = self.base + 'temp/' + run_id + "/"
        self.copyCase(case)
        data = self.runPrimal(initFields, (parameter, nSteps), case)
        shutil.rmtree(case)
        return data

    def adjoint(self, initPrimalFields, paramter, nSteps, initAdjointFields, run_id):
        case = self.base + 'temp/' + run_id + "/"
        self.copyCase(case)
        data = self.runAdjoint(initPrimalFields, (parameter, nSteps), initAdjointFields, case)
        return 


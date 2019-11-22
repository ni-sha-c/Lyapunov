# Want: 1. a curve along the unstable manifold of a point.
# 2. sensitivities of J \circ \varphi_N, for a large N, say k/\lambda_1,
# at points on the curve distributed uniformly on the curve.
from numpy import *
import sys
sys.path.insert(0,'examples/')
from kuznetsov_poincare import Runner
runner = Runner()

# get a point on the attractor.
u0 = runner.u_init 
s0 = 0
u0, obj = runner.primalSolver(u0, s0, 500)

# get CLVs at the point
N = 100
v = rand(3)
v /= linalg.norm(v)
u = copy(u0)
for n in range(N):
    v, sens = runner.tangentSolver(v, u, s0, 1, \
            homogeneous=True)
    v /= linalg.norm(v)
    u, obj = runner.primalSolver(u, s0, 1)
# (u,v) are a fiber in the unstable tangent bundle.




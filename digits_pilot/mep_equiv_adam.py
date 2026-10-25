# Checks that the MEP "RiemannianOptimizer" (base_temp=0) equals Adam without first-moment bias correction,
# i.e. step_MEP(t) = (1 - 0.9^t) * step_Adam(t), on a noisy non-convex objective. NumPy port of both, line for line.
import numpy as np
rng=np.random.default_rng(0)
D=50; A=rng.normal(size=(D,D)); H=A@A.T/D
def grad(p,t): return H@p + 0.5*np.sin(3*p) + 0.1*np.random.default_rng(t).normal(size=D)   # same noise for both runs
def run_mep(steps,lr=1e-2,mass=0.9):
    p=np.ones(D); v=np.zeros(D); h=np.zeros(D); hist=[]
    for t in range(1,steps+1):
        g=grad(p,t); h=0.999*h+0.001*g*g; hh=h/(1-0.999**t); alr=lr/(np.sqrt(hh)+1e-8)
        v=mass*v+(1-mass)*g; step=v*alr; p=p-step; hist.append(step)
    return p,hist
def run_adam(steps,lr=1e-2,b1=0.9,b2=0.999,warm=False,ref=None):
    p=np.ones(D); m=np.zeros(D); v=np.zeros(D); hist=[]
    for t in range(1,steps+1):
        g=grad(p,t); m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g; mh=m/(1-b1**t); vh=v/(1-b2**t)
        step=(lr*(1-b1**t) if warm else lr)*mh/(np.sqrt(vh)+1e-8); p=p-step; hist.append(step)
    return p,hist
pm,hm=run_mep(2000); pa,ha=run_adam(2000); pw,hw=run_adam(2000,warm=True)
print("max |MEP step − Adam step × (1−0.9^t)| over 2000 steps:", max(np.abs(a-b).max() for a,b in zip(hm,hw)))
print("final params, MEP vs Adam-with-(1−0.9^t)-lr: max diff", np.abs(pm-pw).max())
print("first-step ratio MEP/Adam:", (np.abs(hm[0])/np.abs(ha[0])).mean().round(4), " | step 10:", (np.abs(hm[9])/np.abs(ha[9])).mean().round(4), " | step 50:", (np.abs(hm[49])/np.abs(ha[49])).mean().round(4))

import numpy as np
from ..RK import RK4

class L63:
    '''
    Lorenz 63 system with the forcing term F as the only parameter.
    See Lorenz 1963 "Deterministic Nonperiodic Flow."
    '''
    
    def __init__(self):
        pass
                   
    def f(self,x,theta=[10,28,8/3]):        
        dX = np.zeros(3)
        dX[0] = theta[0]*(x[1]-x[0])
        dX[1] = x[0]*(theta[1]-x[2])-x[1]
        dX[2] = x[0]*x[1]-theta[2]*x[2]
        return(dX)
    
    def M(self,x,theta=[10,28,8/3],dt=0.01):
        return(RK4(x,self.f,dt,theta))

    def get_data(self,x0=None,theta=[10,28,8/3],nSteps = 500,dt=0.01):
        
        if x0 is None:
            x0 = np.random.uniform(-5*np.ones(3),5*np.ones(3))
            for _ in range(10000):
                x0 = self.M(x0,theta=theta,dt=dt)

        x_path = np.zeros((3,nSteps))
        x_path[:,0] = np.ravel(x0)

        for kk in range(1,nSteps):
            x_path[:,kk] = self.M(x_path[:,kk-1],theta=theta,dt=dt)
        
        return(x_path)
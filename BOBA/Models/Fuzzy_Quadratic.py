import numpy as np

class FuQu:
    '''
    This is a quadratic function with a added parameter to make it ''fuzzy''.
    This function can be in any dimension.
    '''
    def __init__(self,gamma):
        self.nDim = len(gamma)
        self.gamma = np.array(gamma).reshape((self.nDim,))
    
    
    def eval(self,X):
        if len(X.shape) == 1 and X.shape[0]==self.nDim:
            X=X.reshape((self.nDim,1))
            nObs = 1
        elif len(X.shape) == 2 and X.shape[0]==self.nDim:
            nObs = X.shape[1]
        elif len(X.shape) == 2 and X.shape[1]==self.nDim:
            X = X.T
            nObs = X.shape[1]
        else:
            raise ValueError('gamma has shape (nDim,1) and X must have shape (nDim,), (nDim,nObs), or (nObs,nDim)')
            
        y = np.zeros((nObs,1))
        for kk in range(nObs):
            y[kk,0] = -(np.linalg.norm(X[:,kk])+ 0.5*np.sin(np.dot(self.gamma,X[:,kk])))**2
        return(y)
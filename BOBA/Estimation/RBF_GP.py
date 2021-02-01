import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

class RBF_GP:
    '''
    Radial Basis Function:
    $$ \Sigma(x , x') = \alpha \exp[ -\sum_{k=1}^d \beta_k(x-x')^2   ] $$
    where $(\alpha,\beta_1,...,\beta_d)$ are the kernel's hyperparameters.
    '''
    def __init__(self,X,y,params=None,obs_var = None):
        nSamps = len(y)
        self.y = y.reshape((nSamps,1))
        assert len(X.shape)== 2,\
            'X must have shape (n_Samples , n_Dimension)'
        if X.shape[0] == nSamps:
            self.X = X
        else:
            self.X = X.T
        self.nDim = X.shape[1]
        if params is None:
            self.params = np.ones(self.nDim+1)
            if self.nDim == 1:
                self.params[1:] = np.std(X)
            else:
                self.params[1:] = (np.diag(np.cov(X.T)))**0.5
        else:
            self.params = params
        if obs_var is None:
            self.obs_var = 0
        else:
            self.obs_var = obs_var
        
    def kernel(self,X=None,params = None):
        if X is None:
            X = self.X
        if params is None:
            params = self.params
        assert len(X.shape)==2 and X.shape[1]==self.nDim, \
            'X must have shape (n_Samples , n_Dimension)'
        
        nSamps = X.shape[0]    
        dists = pdist((1/params[1:])*X ,metric='sqeuclidean')
        K = np.exp(-1* dists)
        K = squareform(K)
        np.fill_diagonal(K, 1)
        return(params[0]*K)
    
    def lml(self,params= None):
        if params is None:
            params = self.params
            
        Sigma = self.kernel(params = params)+self.obs_var*np.identity(self.X.shape[0])
        mu_hat = np.sum(np.linalg.solve(Sigma,self.y))
        mu_hat = mu_hat/np.sum(np.ravel(1/Sigma))
        
        lml = 0.5*(self.y-mu_hat).T@np.linalg.solve(Sigma,self.y-mu_hat)
        sgn , det = np.linalg.slogdet(Sigma)
        lml += sgn*0.5*det

        return(lml[0,0])
        
    
    def fit(self,min_method ='L-BFGS-B'):
        bounds = [(-1E-10,np.inf) for kk in range(self.nDim+1)]
        res = minimize(self.lml,self.params,method =min_method,bounds = bounds)

        self.params = res.x
        kernel = ''
        for kk in range(self.nDim):
            kernel  = kernel + '%.3f'%(res.x[kk+1]) +', ' 
        self.kernel_ = '%.3f'%(np.sqrt(res.x[0]))+'**2 * RBF( length_scale = [' + kernel + '] )'
        if res.success is False:
            print('Fit fails to converge.')
            print(res.message)
    
    def predict(self, X_pred, return_std=False, return_cov=False):
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        assert len(X_pred.shape)==2 and X_pred.shape[1]==self.nDim, \
            'X_pred must have shape (n_Samples , n_Dimension)'
            
        nPred = X_pred.shape[0]
        nSamps = self.X.shape[0]
        mu = np.mean(self.y)*np.ones((nPred,1))
        
        Sigma = self.kernel(X=np.concatenate((self.X,X_pred)))

        mu_n = mu + Sigma[nSamps:,:nSamps]@np.linalg.solve(Sigma[:nSamps,:nSamps]+self.obs_var*np.identity(nSamps),self.y-mu[0,0])
        
        Sigma_pred = Sigma[nSamps:,nSamps:] - Sigma[nSamps:,:nSamps]@np.linalg.solve(Sigma[:nSamps,:nSamps]+self.obs_var*np.identity(nSamps),Sigma[:nSamps,nSamps:])
        
        if return_std:
            return(mu_n,np.diag(Sigma_pred).reshape(nPred,1)**0.5)
        elif return_cov:
            return(mu_n,Sigma_pred)
        else:
            return(mu_n)
        
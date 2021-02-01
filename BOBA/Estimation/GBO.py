import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class GBO:
    '''
    See Frazier and Wang (2015) Bayesian optimization for materials design
    
    :param func:
        The cost function to be maximized in the form y* = func(x^*)
    :param nDim:
        Dimension of func's input variable
    :param X:
        array like with shape (nObs,nDim)
        Current states where we have observed self.func
    :param y:
        array line with shape (nObs,1)
        Current observations of self.func, i.e., y = self.func(X)
    :param bounds:
        array line with shape (nDim,2)
        lower bound and upper bound for parameters
    '''
    
    def __init__(self,func,nDim,X,y,bounds,obs_var = None):
        self.func = func
        self.nDim = nDim
        assert np.array(X).shape[1] == nDim ,\
            'X must have shape (nObs,nDim)'
        self.X = np.array(X)
        assert np.array(y).shape == (np.array(X).shape[0],1) ,\
            'y must have shape (nObs,1) where X.shape = (nObs,nDim)'
        self.y = np.array(y)
        assert np.array(bounds).shape == (nDim,2),\
            'bounds must have shape (nDim,2)'
        self.bounds = np.array(bounds)
        self.obs_var = obs_var
        
    def neg_EI(self,x):
        # Returns negative expected improvement for minimization algo
        x = x.reshape((1,self.nDim))
        f_star = np.max(self.y)
        mu_x , std_x = self.GP.predict(x,return_std=True)
        return(-(((mu_x - f_star)*norm.cdf((mu_x-f_star)/std_x)+
                  std_x*norm.pdf((mu_x-f_star)/std_x))[0,0]))
    
    
    def Expected_Improvement(self,min_method ='L-BFGS-B',nx0 = 20):
        '''
        :param min_method:
            scipy.optimize minimization method to minimize the expected imporvement function
            ********
            I should add the option to use the EI gradient
            ********
        :param nx0:
            The number of samples used to find a good initial condition for the minimizer
        '''
        if not self.obs_var is None:
            print('WARNING: Expected Improvement is not designed for noisy observations. Consider using the Knowledge Gradient.')
        
        if not hasattr(self,'GP'):
            raise AttributeError('Must define fitted Gaussian process')
        
        def neg_GP_mean(x):
            return(-self.GP.predict(x.reshape((1,self.nDim))))
        
        EI0_best = np.inf
        for kk in range(nx0):
            x0 = np.random.uniform(self.bounds[:,0],self.bounds[:,1])
            try:
#                print('0: ',x0)
                res = minimize(neg_GP_mean,x0,method = min_method,bounds = self.bounds)
#                print('1: ', res.x)
                res = minimize(self.neg_EI,res.x,method = min_method,bounds = self.bounds)
#                print('2: ', res.x)
                if res.fun < EI0_best:
                    x_best = res.x
                    EI0_best = res.fun
            except Warning:
                'WARNING:\n'
                print(res.message)

        x = x_best.reshape((1,self.nDim))
        check = np.array([np.linalg.norm(self.X[kk,:]-x)<1E-8 for kk in range(len(self.X))])
#        if np.any(check):
#            print('Expected Improvement did not select a new sample.')
#        else:
        self.X = np.concatenate((self.X,x))
        self.y = np.concatenate((self.y,self.func(x).reshape((1,1))))
        
    def neg_KG_func(self,x):
        A_n = self.X
        A_np1 = np.concatenate((self.X,np.asarray(x).reshape((1,self.nDim))))
        
        a = self.GP.predict(A_np1)
        mu_n = a
        b = self.GP.kernel_(A_np1)[:,-1]#-self.obs_var
        b /= np.sqrt(b[-1])
        
        a = np.array(a).ravel()
        b = np.array(b).ravel()
        Index = np.argsort(b)
        a = a[Index]
        b = b[Index]
        Index = []
        kk = 0
        for _ in range(len(b)-1):
            if np.abs(b[kk] - b[kk+1])<=1E-5:
                if a[kk] < a[kk+1]:
                    a = np.delete(a,[kk])
                    b = np.delete(b,[kk])
                else:
                    a = np.delete(a,[kk+1])
                    b = np.delete(b,[kk+1])
            else:
                kk += 1        
        
        M = len(a)
        c = np.zeros(M)
        
        c[0] = np.inf
        A = [0]
        for ii in range(M-1):
            c[ii+1] = np.inf
            loop_go = True
            while loop_go:
                jj = A[-1]
                c[jj] = (a[jj]-a[ii+1])/(b[ii+1]-b[jj])
                if len(A)>1:
                    kk = A[-2]
                    if c[jj] <= c[kk]:
                        A = A[:-1]
                    else:loop_go = False
                else:loop_go = False
            A.append(ii+1)
        a = a[A]
        b = b[A]
        c = c[A]
        M = len(A)
        
        h = [(b[kk+1]-b[kk])*(norm.pdf(-np.abs(c[kk])) - np.abs(c[kk])*norm.cdf(-np.abs(c[kk]))) for kk in range(M-1)]
        return(-np.sum(h))
    
    def Knowledge_Gradient(self,min_method ='L-BFGS-B',nx0 = 5):
        if not hasattr(self,'GP'):
            raise AttributeError('Must define fitted Gaussian process')
        
        def neg_GP_mean(x):
            return(-self.GP.predict(x.reshape((1,self.nDim))))
        
        KG0_best = np.inf
        for kk in range(nx0):
            x0 = np.random.uniform(self.bounds[:,0],self.bounds[:,1])
            try:
#                print('0: ',x0)
                res = minimize(neg_GP_mean,x0,method = min_method,bounds = self.bounds)
#                print('1: ', res.x)
                res = minimize(self.neg_KG_func,res.x,method = min_method,bounds = self.bounds)
#                print('2: ', res.x)
                if res.fun < KG0_best:
                    x_best = res.x
                    KG0_best = res.fun                   
            except Warning:
                'WARNING:\n'
                print(res.message)
            
        x = x_best.reshape((1,self.nDim))
        check = np.array([np.linalg.norm(self.X[kk,:]-x)<1E-8 for kk in range(len(self.X))])
        if np.any(check):
            print('Knowledge Gradient did not select a new sample.')

        self.X = np.concatenate((self.X,x))
        self.y = np.concatenate((self.y,self.func(x).reshape((1,1))))
        
        
    
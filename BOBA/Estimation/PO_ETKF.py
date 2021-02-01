import numpy as np
   
class PO_ETKF:
    '''
    Perturbed Observation Ensemble Transform Kalman filter
    See "Adaptive Sampling with the Ensemble Transform Kalman Filter." CRAIG H. BISHOP, et al., 2001
    
    This implementation assumes the observation matrix / function is built into the forward model M.
    
    :param M:
        The model that advances the state one step in time, the only imput should be the current state x.
        The output of M shoud be the observation of the forward model Hx (using classical Kalman filter notation) 

    '''
    def __init__(self,nDim,H=None,sqrt_R=None):
        '''
        :param nDim:
            The observation and state dimension (These must be the same since no obsevartion matrix / function is used)
        :param H:
            Observation matrix
        :param sqrt_R:
            Squareroot of the observation covariance matrix.
            sqrt_R must be a diagonal matrix so that 
            $$(\tilde{H})^T = (R^{-0.5}H)^T = H^T(R^{-0.5})^T = H^T(R^{-0.5}) = \tilde{H}^T$$.
            See above reference, eqs. 8.
        '''
        self.nDim = nDim
        if H is None:
            self.H = np.identity(nDim)
        else:
            self.H = H
        self.obs_Dim = self.H.shape[0]
        # Form the observation error covariance matrix
        if sqrt_R is None:
            self.sqrt_R = np.identity(self.obs_Dim)
        else:
            assert np.all(sqrt_R == np.diag(np.diag(sqrt_R))), \
                "sqrt_R must be a diagonal matrix"
            self.sqrt_R = sqrt_R
                
    def run(self,x_f,y,check_P_a = False):
        '''
        :param x_f:
            Forecast ensemble with shape = (State dimension , Number of Ensemble members)
        :param y:
            Observation with shape = (Observation dimesnion , )
        :param check_P_a:
            True / False: For comparison, output the theoretical and the sample analysis covariance.
        '''
        self.nE = x_f.shape[1]
        y = y.reshape((self.nDim,))
        # ensemble mean for each parameter
        mu = np.mean(x_f,axis=1)
        # Ensemble of forward observations
        Hx = H@x_f
        # Mean of prior in obs space
        mu_Hx = np.mean(Hx,axis=1)
        # Parameter ensemble perturbations
        Xp = x_f - np.tile(mu,(self.nE,1)).T
        # Fwd obs perturbations -- mu_Hx is the mean of the fwd obs ensemble
        HXp = Hx - np.tile(mu_Hx,(self.nE,1)).T     
        # Add perturbed observations
        # Create an ensemble of observations by drawing from N(0,1) and modifying to be
        # consistent with the observation error variance.
        # Ensemble perturbations
        yp = self.sqrt_R@np.random.randn(self.nDim,self.nE)
        # mean of perturbed obs ensemble
        yp_mu = np.mean(yp,axis=1)
        # Subtract off ensemble mean
        yp = yp - np.tile(yp_mu,(self.nE,1)).T
        # Obs operator perturbation matrix from Bishop et al., 2001
        Z = HXp / np.sqrt(self.nE-1)
        # Get eigenvalues of the normalized obs operator perturbation matrix
        A_tmp = np.linalg.solve(self.sqrt_R,Z)
        E,S,Ct = np.linalg.svd(A_tmp,full_matrices=False)
        C = Ct.T
        # Note that Gamma output from svd contains the singular values
        # the square root of the Gamma matrix in Bishop et al 2001. 
        # Square Gamma to get the eigenvalue matrix
        Gamma = np.diag(S**2)
        # Compute the gain
#####  Derek's Code has Gamma+np.identity(self.nDim) but this is a problem when nDim > nE
        Za = (np.identity(self.nE) - C@Gamma@(np.linalg.solve(Gamma+np.identity(Gamma.shape[0]),C.T)))@A_tmp.T
#####
        # Kalman Gain: (nDim x nE) * (nE x nDim) * (nDim x nDim) = (n x nDim)
        K = ( Xp/np.sqrt(self.nE-1) ) @np.linalg.solve(self.sqrt_R.T,Za.T).T
        # Create posterior ensemble that includes perturbed obs
        
        X_a = Hx+K@(yp+np.tile(y,(self.nE,1)).T-(HXp+np.tile(mu_Hx,(self.nE,1)).T ))
        
        if check_P_a:
            # Check if ensemble forecast covariance is similar to theorietical forecast cov.
            P_a_ensemble = np.cov(X_a)
            T = C@(np.diag((np.diag(Gamma)+1)**-0.5))
            P_a_theory = Z@T@T.T@Z.T
            return(X_a,P_a_ensemble,P_a_theory)
        else:
            return(X_a)
        
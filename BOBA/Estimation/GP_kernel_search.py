import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (DotProduct,RBF, Matern,WhiteKernel\
                 ,ExpSineSquared,ConstantKernel,RationalQuadratic)
from sklearn.gaussian_process.kernels import Sum as kernsum
from sklearn.gaussian_process.kernels import Product as kernprod

class GP_kernel_search:
    def __init__(self,X,y,n_lvls,Words, first_Words = None,print_output = False):
        self.X = X
        self.y = y
        self.n_lvls = n_lvls
        self.Words = Words
        self.first_Words = first_Words
        self.print_output = print_output
        
    def BIC(self,model):
        '''
        Parameters
        ----------
        model: Gaussian Process

        Output
        ------
        BIC_val: float
            Bayesian Information Criterion
            BIC_val = \log( p(D|M) ) - 0.5*|M|*\log(N)
            p(D|M) is the optimized log marginal likelihood of the data
            evaluated at the optimized kernel parameters, |M| is the 
            number of free kernel parameters and N is the number of data
            points.  For reference, see 
                Duvenaud, D., Lloyd, J., Grosse, R., Tenenbaum, J. & Zoubin
                , G.. (2013). Structure Discovery in Nonparametric Regression
                through Compositional Kernel Search. Proceedings of the 30th
                International Conference on Machine Learning, in PMLR 28(3):
                1166-1174
        '''
        lml = -model.log_marginal_likelihood_value_
        params = model.kernel_.theta
        BIC_val = lml-0.5*(len(params))*np.log(len(self.X))
        return(BIC_val)

#--------------------------------------------------------------------#    
    def get_possible_kernels(self,last_kernel = None ):
        '''
        Parameters
        ----------
        last_kernel: kernel object
            The optimized kernel from the previous step for which the next set
            of kernels will be built from Words.

        Outputs
        -------
        Kernels: List
            The list of the next row of kernels in the kernel search    
        '''
        Kernels = []
        for word in self.Words:
            Kernels.append(kernprod(last_kernel,word))
            Kernels.append(kernsum(last_kernel,word))
        return(Kernels)

#--------------------------------------------------------------------#

    def get_next_kernel(self,Kernels):
        ''' 
        Parameters
        ----------
        Kernels: List of kernel objects
            Kerenels to be measured by choice_func

        Outputs
        -------
        Next_kernel: kernel object
            Kernel with the highest choice_val and will be used to generate the 
            next row of kernels in the kernel search
        '''
        import warnings
        with warnings.catch_warnings():
            choice_vals = []
            for kern in Kernels:
                if self.print_output:
                    print('            |             |',kern)
                model = GPR(kernel=kern,normalize_y=True,n_restarts_optimizer=25)
                try:
                    model.fit(self.X,self.y)
                    tmp_val = self.BIC(model)
                    if self.print_output:
                        print('            |             | BIC: ',tmp_val)
                    choice_vals.append([model.kernel_,tmp_val])
                except: pass
            if choice_vals == []: raise ValueError('No kernel could be fit.')
            else:
                _index = np.argmax(list(np.array(choice_vals)[:,1]))
                Next_kernel = np.array(choice_vals)[_index,0]
                return(Next_kernel)

#--------------------------------------------------------------------#
    def comp_kernel_search(self):
        '''
        Output
        ------
        last_kernel: kernel object
            Best kernel based on n_lvls, start_lvl and choice_func
        '''
        for kk in range(self.n_lvls):
            if self.print_output:
                print('            | Running kernel searching algorithm:')
                print('            |')
                print('            | Level '+str(kk+1)+'\n')
            if kk == 0:
                if self.first_Words is None: Kernels = [] + self.Words
                else: Kernels = self.first_Words
            else:
                Kernels = []
                for word in self.Words:
                    Kernels.append(kernprod(last_kernel,word))
                    Kernels.append(kernsum(last_kernel,word))

            last_kernel = self.get_next_kernel(Kernels)
            if self.print_output:
                print('            | Selected kernel:')
                print('            | ', last_kernel)
        return(last_kernel)


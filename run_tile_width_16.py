import pickle
import numpy as np
from scipy.special import lambertw
import emcee
from scipy.ndimage.filters import gaussian_filter1d as gf1d

file = open('./file_ID.txt', 'r') 
file_ID = file.read()
index = file_ID.find('\n')
file_ID = file_ID[:index] + '_TW16' 

tile_width = 16

nSteps = 1E5

fileObject1 = open('./Pickled_Clouds/%s_mean'%(file_ID),'rb') 
cloud_H =pickle.load(fileObject1) 

fileObject1 = open('./Pickled_Clouds/%s_cov'%(file_ID),'rb') 
cloud_Cov =pickle.load(fileObject1) 

fileObject1 = open('./Pickled_Clouds/MCMC_initial_ensemble','rb') 
ensemble =np.array(pickle.load(fileObject1) )


from Centered_Mean import MCMC_model
from Centered_Mean import RealBeta
from Centered_Mean import lnprob

Probs = []
for theta in ensemble:
    Probs.append(lnprob(theta , cloud_H,cloud_Cov))
ndim ,nwalkers = 4 , 20
pos = [ensemble[kk,:].ravel() for kk in np.argsort(Probs)[-nwalkers:] ]


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(cloud_H,cloud_Cov))

sampler.run_mcmc(pos,nSteps)

fileMCMC = open('%s_MCMC_chain'%(file_ID),'wb')
pickle.dump(sampler.chain,fileMCMC)
fileMCMC.close()

fileMCMC = open('%s_MCMC_lnprob'%(file_ID),'wb')
pickle.dump(sampler.lnprobability,fileMCMC)
fileMCMC.close()


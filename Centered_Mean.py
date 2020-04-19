# Adaptive ODE Solve time
# [H0, tau, T, alhpa] space

import pickle
import numpy as np
from scipy.special import lambertw
from scipy.ndimage.filters import gaussian_filter1d as gf1d


def dim_model(H_hist,H_eval,t,H_args):
    N = 25E6   # m^-3
    H0, tau, T , Hint, alpha, dt = H_args
    if T == 0: HT=H_eval
    else: 
        past_time = int((t-T)/dt)
        HT = H_hist[past_time]
    return((H0-H_eval)/tau - alpha*HT**2/np.sqrt(N))

def check_period(H,tol):
    dH = gf1d(H[1:]-H[:-1],2)
    Index1 = [dH[kk-1]<0 and 0<dH[kk+1] for kk in range(1,len(dH)-1)]
    Index1 = np.where(np.array(Index1) == True)[0]
    dH = np.abs(dH)
    Index2 = [dH[kk-1]>dH[kk] and dH[kk]<dH[kk+1] for kk in range(1,len(dH)-1)]
    Index2 = np.where(np.array(Index2) == True)[0]
    Index = [val+1 for val in Index2 if val in Index1]
    cycle1 = H[Index[-3]:Index[-2]]
    cycle2 = H[Index[-2]:Index[-1]]
    l = np.min([len(cycle1),len(cycle2)])
    cycle1 = cycle1[:l]
    cycle2 = cycle2[:l]
    scale = np.max(cycle1)-np.min(cycle1)
    rmse = np.sqrt(np.mean((cycle1-cycle2)**2))
    if rmse < np.max([tol*scale,1]):
        return(True,Index[-2],Index[-1])
    else:
        return(False,np.NaN,np.NaN)

def dim_RK4(H0, tau, T, alpha, Hint,solve_time,dt):
    H_args = H0, tau, T , Hint, alpha, dt
    odetime = np.arange(0,solve_time+T,dt)
    Y = np.zeros(len(odetime))
    past_steps = len(np.arange(0,T+dt,dt))
#    print(Hint[-past_steps:])
    for kk , time in enumerate(np.arange(0,T+dt,dt)):
        Y[kk] = Hint[kk-past_steps]  
    for jj , t_n in enumerate(odetime):
        if t_n < T+dt: pass
        else:
            y_n = Y[jj-1]
            k1 = dim_model(Y,y_n,t_n,H_args)
            y_eval = y_n + dt*k1/2.0
            k2 = dim_model(Y,y_eval,t_n+dt/2,H_args)
            y_eval = y_n + dt*k2/2.0
            k3 = dim_model(Y,y_eval,t_n+dt/2,H_args)
            y_eval = y_n + dt*k3
            k4 = dim_model(Y,y_eval,t_n+dt,H_args)
            Y[jj]=y_n+dt*(k1+2*k2+2*k3+k4)/6.0
    return(Y)


def MCMC_model(H0, tau, T, alpha,Z,solve_time = 1):
    mid_cloud = np.argmax(Z)
    
    nn = 10
    nsteps = 1440*nn
    dt = 1/nsteps
    
    period_check = False
    Hint = np.zeros(int(T*nsteps)+2)+Z[0]
    
    
    count = 0
#    solve_time = 1
    tol=0.001
    while not period_check:
        M_theta = dim_RK4(H0, tau, T, alpha, Hint ,solve_time=solve_time,dt=dt)
        period_check, t1,t2 = check_period(M_theta,tol)
        count +=1
        solve_time+=1
        if 5<count<10:
            tol = 0.005
        if count > 10:
            tol += 1
            print('!!!!!!!! ',tol,count,' : ', H0, tau, T, alpha)
    t1 ,t2 = t1//nn,t2//nn
    
    index = [nn*kk for kk in range(len(M_theta)//nn)]
    M_theta = M_theta[index]
    M_theta = M_theta[t1:t2]

    
    tmid = np.argmax(M_theta)
    Model_cloud = np.zeros_like(Z)

    Model_cloud[mid_cloud]=M_theta[tmid]

    # Left arm
    if tmid < mid_cloud:
        for kk in range(1,tmid+1):
            Model_cloud[mid_cloud-kk] = M_theta[tmid-kk]
    if tmid == mid_cloud:
        Model_cloud[:mid_cloud] = M_theta[:tmid]
    if tmid > mid_cloud:
        for kk in range(1,mid_cloud+1):
            Model_cloud[mid_cloud-kk] = M_theta[tmid-kk]

    #Right arm
    arm1 = len(Model_cloud[mid_cloud:])
    arm2 = len(M_theta[tmid:])
    if arm2 < arm1:
        for kk in range(arm2):
            Model_cloud[mid_cloud+kk] = M_theta[tmid+kk]
    if arm2 == arm1:
        Model_cloud[mid_cloud:] = M_theta[tmid:]
    if arm2 > arm1:
        for kk in range(arm1):
            Model_cloud[mid_cloud+kk] = M_theta[tmid+kk]  
    return(Model_cloud)

def hsts(mu): return(np.sqrt(mu**2/4+mu)-mu/2)

def RealBeta(mu,D):
    return(np.real(1/D*lambertw(-2*hsts(mu)/mu*D*np.exp(D))-1))


def lnlike(M_theta,Z,cloud_Cov):
    _l = len(Z)
    Z = Z.reshape((_l,1))
    M_theta = M_theta.reshape((_l,1))

    tmp = -0.5*(Z-M_theta).T@(np.linalg.solve(cloud_Cov,Z-M_theta))
    if tmp==np.NaN : raise ValueError(M_theta.shape, M_theta) 
    return(tmp[0][0])

def cloud_2_stats(M_theta):
    _l = len(M_theta)
    M_theta = M_theta.reshape((_l,))
    
    try: t1 = np.min(np.where(M_theta>0))
    except: t1=0
        
    try: t2 = np.max(np.where(M_theta>0))
    except: t2= len(M_theta)
        
    tmid = np.argmax(M_theta)
    per = t2-t1
    amp = M_theta[tmid] - M_theta[t1]
    growt = tmid - t1
    decayt = t2-tmid
    
    stat = np.array([per,amp, growt,decayt]).reshape((4,1))
    return(stat)

def stat_lnlike(M_theta,Z,stat_mean,stat_Cov):
    _l = len(Z)
    Z = Z.reshape((_l,1))
    M_theta = M_theta.reshape((_l,))

    stat = cloud_2_stats(M_theta)
    stat_mean = stat_mean.reshape((4,1))
    
    tmp = -0.5*(stat-stat_mean).T@(np.linalg.solve(stat_Cov,stat-stat_mean))
    if tmp==np.NaN : raise ValueError(M_theta.shape, M_theta) 
    return(tmp[0][0])


def lnprior(theta,Z):
    H0, tau, T, alpha = theta
    
    mu = np.sqrt(25E6)/(alpha*tau*H0)
    D = T/tau
    
    ###   mu    , tau,    D, alpha
    lb = [0     , 0,      0, 50 ]
    ub = [0.4153, 24/144, 1, 1500]
    
    theta_check = mu    , tau,    D, alpha
    
    if all(lb[kk]<theta_check[kk]<ub[kk] for kk in range(4)):
        Re = RealBeta(mu,D)
        if 0 < Re < 0.5:
            try: 
                M_theta = MCMC_model(H0, tau, T, alpha,Z)
            except: return([])
            if np.all(M_theta >= 0):
                return(M_theta)
            else:
                return([])
        else:
            return([])
    else:
        return([])
    
def lnprob(theta , Z,cloud_Cov ):
    lp = lnprior(theta,Z)
    if len(lp) == 0:
        return(-np.inf)
    else:
        return(lnlike(lp,Z,cloud_Cov))
    
    
    
def stat_lnprob(theta,Z,stat_mean,stat_Cov):
    lp = lnprior(theta,Z)
    if len(lp) == 0:
        return(-np.inf)
    else:
        return(stat_lnlike(lp,Z,stat_mean,stat_Cov))

##############################################################################
##############################################################################
##############################################################################
# Process the LES 

def get_X_ave(X,Tile_width,threshold_parameter,threshold_value=0):
    nMin = X.shape[0]
    nTiles = int(X.shape[1]/Tile_width)
    
    X_ave = np.zeros((nMin,nTiles,nTiles))
    
    for kk in range(nMin):
        index = threshold_parameter[kk,:,:]>threshold_value
        for jj in range(nTiles):
            for mm in range(nTiles):
                
                x1 = Tile_width*(jj)
                x2 = Tile_width*(jj+1)

                y1 = Tile_width*(mm)
                y2 = Tile_width*(mm+1)
                count = sum(sum(index[x1:x2,y1:y2]))
                if count != 0: X_ave[kk,jj,mm] = sum(sum(index[x1:x2,y1:y2]*X[kk,x1:x2,y1:y2]))/count
    return(X_ave)

def save_tile_width_nc_file(X,file_name,LES_nc,threshold_parameter,threshold_value=0,Powers = [0,1,2,3,4,5,6,7,8,9]):
    from netCDF4 import Dataset
    dataset = Dataset(file_name,'w',format='NETCDF4_CLASSIC')
    x = dataset.createDimension('x',X.shape[1])
    y = dataset.createDimension('y',X.shape[2])
    time = dataset.createDimension('time',X.shape[0])

    dataset.createVariable('time',np.float64,('time',))
    dataset['time'][:]=LES_nc['time'][:]
    for power in Powers: 
        print('Creating tile width ',str(2**power))
        X_tmp = np.zeros(X.shape)
        Tile_width = 2**power


        variable_name = 'Tile Width %s'%(Tile_width)
        var = dataset.createVariable(variable_name,np.float64,('time','y','x'))

        nTiles = int(X.shape[1] / Tile_width)
        X_ave= get_X_ave(X,Tile_width,threshold_parameter,threshold_value)
        for kk in range(nTiles):
            for jj in range(nTiles):
                tmp = np.repeat(np.repeat(X_ave[:,kk,jj][:,np.newaxis],Tile_width,axis = 1)[:,:,np.newaxis],Tile_width,axis=2)
                X_tmp[:,Tile_width*kk:Tile_width*(kk+1),Tile_width*jj:Tile_width*(jj+1)]=tmp
        dataset[variable_name][:]=X_tmp
    dataset.close()
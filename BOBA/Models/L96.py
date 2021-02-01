import numpy as np
#from matplotlib.widgets import Slider, Button, RadioButtons
#import matplotlib.pyplot as plt
from ..RK import RK4

class L96_1:
    '''
    Lorenz 96 single scale system.
    This system is listed in Eq. (1) of Lorenz 1996 "Predictability, a problem partial solved."
    
    :param nK:
        The number of states X.
    '''
    
    def __init__(self,nK):
        self.nK = nK
        assert self.nK > 4, \
            "Must have more than four X states."
                   
    def f(self,x,theta=8):
        dX = np.zeros(self.nK)
        
        for kk in range(self.nK):
            dX[kk] = -x[kk-2]*x[kk-1]+x[kk-1]*x[(kk+1)%self.nK]-x[kk]+theta
        return(dX)
    
    def M(self,x,theta=8,dt=0.01):
        return(RK4(x,self.f,dt,theta))

    def get_data(self,x0=None,theta=8,nSteps = 50,dt=0.01):
        
        if x0 is None:
            x0 = np.random.uniform(-5*np.ones(self.nK),5*np.ones(self.nK))            
            for _ in range(1000):
                x0 = self.M(x0,theta=theta,dt=dt)

        x_path = np.zeros((self.nK,nSteps))
        x_path[:,0] = np.ravel(x0)

        for kk in range(1,nSteps):
            x_path[:,kk] = self.M(x_path[:,kk-1],theta=theta,dt=dt)
        return(x_path)

    def get_L(self,loc_val):
        L = np.zeros((self.nK,self.nK))
        for kk in range(self.nK):
            for jj in range(kk,self.nK):
                tmp = np.min([(jj-kk)%self.nK,np.abs(self.nK -(jj-kk)%self.nK )])
                L[kk,jj] = np.exp(-tmp**2/(2*loc_val**2))
                L[jj,kk] = L[kk,jj]        
        return(L)

class L96_23:
    '''
    Lorenz 96 coupled fast/slow system.
    This system are listed in Eqs. (2) and (3) of Lorenz 1996 "Predictability, a problem partial solved."
    Also see: Schneider, T., Lan, S., Stuart, A., & Teixeira, J. (2017). Earth system modeling 2.0
    
    :param nK:
        The number of 'slow' states X.
    :param nJ:
        The number of 'fast' states Y coupled to each X state.
    '''
    
    def __init__(self,nK,nJ):
        self.nK = nK
        assert self.nK > 4, \
            "Must have more than four X states."
        self.nJ = nJ
        assert self.nJ > 4, \
            "Must have more than four Y states."
        self.nDim = nK*(1+nJ)

    def X_Y_to_XY(self,X,Y):
        
        nSteps = X.shape[1]
        
        XY = np.zeros((self.nK+self.nK*self.nJ,nSteps))
        XY[:self.nK,:] = X
        for kk in range(self.nK):
            XY[self.nJ*kk+self.nK:self.nJ*(kk+1)+self.nK,:] = Y[:,kk,:]
        return(XY)

    def XY_to_X_Y(self,XY):

        nSteps = XY.shape[1]
        
        X = np.zeros((self.nK,nSteps))
        Y = np.zeros((self.nJ,self.nK,nSteps))

        X = XY[:self.nK,:]
        for kk in range(self.nK):
            Y[:,kk,:] = XY[self.nJ*kk+self.nK:self.nJ*(kk+1)+self.nK,:]
        return(X,Y)
                   
    def f(self,x,theta=[10,10,10,1]):
        b,c,F,h = theta 
        
        X = x[:self.nK]
        Y = x[self.nK:]
        
        dX = np.zeros(self.nK)
        dY = np.zeros((self.nJ*self.nK))
        
        for kk in range(self.nK):
            dX[kk] = -X[(kk-1)%self.nK]*(X[(kk-2)%self.nK]-X[(kk+1)%self.nK])-X[kk]-h*c*np.mean(Y[kk*self.nJ:(kk+1)*self.nJ])+F
            for jj in range(kk*self.nJ,(kk+1)*self.nJ):
                dY[jj]= -c*b*Y[(jj+1)%(self.nJ*self.nK)]*(Y[(jj+2)%(self.nJ*self.nK)]-Y[(jj-1)%(self.nJ*self.nK)])-c*Y[jj]+c*h*X[kk]
        return(np.concatenate((dX,dY)))
    
    
    def M(self,x,theta=[10,10,10,1],dt=0.001):
        return(RK4(x,self.f,dt,theta))

    def get_data(self,x0=None,theta=[10,10,10,1],nSteps = 500,dt=0.001):
        
        if x0 is None:
            X = np.random.uniform(-1*np.ones(self.nK),5*np.ones(self.nK))
            Y = np.random.randn(self.nJ*self.nK)
            x0 = np.concatenate((X,Y))
            
            for _ in range(10000):
                x0 = self.M(x0,theta=theta,dt=dt)

        x_path = np.zeros((self.nK+self.nK*self.nJ,nSteps))
        x_path[:,0] = np.ravel(x0)

        for kk in range(1,nSteps):
            x_path[:,kk] = self.M(x_path[:,kk-1],theta=theta,dt=dt)
        
        return(x_path)

    def get_L(self,loc_val):
        L = np.zeros((self.nDim,self.nDim))
        for kk in range(self.nK):
            for jj in range(self.nK):
                p1 = np.array((np.cos(2*np.pi*kk/self.nK),np.sin(2*np.pi*kk/self.nK)))
                p2 = np.array((np.cos(2*np.pi*jj/self.nK),np.sin(2*np.pi*jj/self.nK)))

                L[kk,jj] = np.exp(-np.linalg.norm(p1-p2)/loc_val)
                L[jj,kk] = L[kk,jj]

        for k1 in range(self.nK):
            for k2 in range(self.nK):
                L[k1,k2*self.nJ+self.nK:(k2+1)*self.nJ+self.nK]+=L[k1,k2]
        L[self.nK:,:self.nK] = L[:self.nK,self.nK:].T

        for kk in range(self.nK,self.nDim):
            for jj in range(self.nK,self.nDim):
                p1 = np.array((np.cos(2*np.pi*(kk-self.nK)/self.nDim),np.sin(2*np.pi*(kk-self.nK)/self.nDim)))
                p2 = np.array((np.cos(2*np.pi*(jj-self.nK)/self.nDim),np.sin(2*np.pi*(jj-self.nK)/self.nDim)))

                L[kk,jj] = np.exp(-np.linalg.norm(p1-p2)/loc_val)
                L[jj,kk] = L[kk,jj]            
        
        return(L)
    
#    def notebook_plot(self,x_path,dT):
#        X = x_path[:self.nK,:]
#        Y = x_path[self.nK:,:]
#
#        theta_X = np.linspace(0, 2*np.pi, self.nK+1)
#        theta_Y = np.linspace(0, 2*np.pi, self.nJ*self.nK+1)
#
#        def signal_X(kk):
#            Xplt = [val for val in X[:,int(kk/dT)]]+[X[0,int(kk/dT)]]
#            return(Xplt)
#        def signal_Y(kk):
#            Yplt = [val+14 for val in Y[:,int(kk/dT)]]+[Y[0,int(kk/dT)]+14]
#            return(Yplt)
#
#        fig = plt.figure()
#        ax = plt.subplot(111, projection='polar')
#        ax.set_yticks([])
#        ax.set_xticks([])
#
#
#        # Adjust the subplots region to leave some space for the sliders and buttons
#        fig.subplots_adjust(left=0.25, bottom=0.25)
#
#
#        # Draw the initial plot
#        # The 'line' variable is used for modifying the line later
#        [line2] = ax.plot(theta_X,signal_X(-1))
#        [line1] = ax.plot(theta_X,signal_X(-1),'o')
#        [line3] = ax.plot(theta_Y,signal_Y(-1))
#
#        # Add sliders for tweaking the parameters
#
#        # Define an axes area and draw a slider in it
#        slider_ax  = fig.add_axes([0.25, 0.14, 0.65, 0.02])
#        slider = Slider(slider_ax, 'time', 0, X.shape[1]*dT, valinit=0)
#
#        # Define an action for modifying the line when any slider's value changes
#        def sliders_on_changed(val):
#           data_X = signal_X(slider.val)
#            data_Y = signal_Y(slider.val)
#            ax.set_yticks([])
#            ax.set_xticks([])
#            line1.set_ydata(data_X)
#            line2.set_ydata(data_X)
#            line3.set_ydata(data_Y)
#            fig.canvas.draw_idle()
#        slider.on_changed(sliders_on_changed)
#
#        # Add a button for resetting the parameters
#        reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
#        reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.975')
#        def reset_button_on_clicked(mouse_event):
#            slider.reset()
#        reset_button.on_clicked(reset_button_on_clicked)
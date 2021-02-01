def RK4(x,f,dt,args):
    # Runge-Kutta 4
    X_k1 = f(x,args)
    X_k2 = f(x+dt*X_k1/2,args)
    X_k3 = f(x+dt*X_k2/2,args)
    X_k4 = f(x+dt*X_k3,args)

    X = x+dt*(X_k1+2*X_k2+2*X_k3+X_k4)/6
    return(X)
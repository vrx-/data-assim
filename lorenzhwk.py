import numpy as np
import matplotlib.pyplot as plt


from __future__ import print_function


class L40(object):
    '''Lorenz 40 model of zonal atmospheric flow'''
    
    def __init__(self, members=1, n=40, dt=0.05, F=8):
        self.n = n
        self.dt = dt
        self.dtx = dt
        self.x = np.random.normal(0., 0.1, size=(members, n))
        self.members = members
        self.F = F
        self.his = []
        self.var = []
    
    def dxdt(self):
        dxdt = np.zeros((self.members, self.n),'f8')
        for n in range(2,self.n-1):
            dxdt[:,n] = -self.x[:,n-2]*self.x[:,n-1] +  \
                        self.x[:,n-1]*self.x[:,n+1] - self.x[:,n] + self.F
        dxdt[:,0] = -self.x[:,self.n-2]*self.x[:,self.n-1] +  \
                self.x[:,self.n-1]*self.x[:,1] - self.x[:,0] + self.F
        dxdt[:,1] = -self.x[:,self.n-1]*self.x[:,0] + \
                self.x[:,0]*self.x[:,2] - self.x[:,1] + self.F
        dxdt[:,self.n-1] = -self.x[:,self.n-3]*self.x[:,self.n-2] + \
                            self.x[:,self.n-2]*self.x[:,0] - \
                            self.x[:,self.n-1] + self.F
        return dxdt
    
    def rk4step(self):
        h = self.dt; hh = 0.5*h; h6 = h/6.
        x = self.x
        dxdt1 = self.dxdt()
        self.x = x + hh*dxdt1
        dxdt2 = self.dxdt()
        self.x = x + hh*dxdt2
        dxdt = self.dxdt()
        self.x = x + h*dxdt
        dxdt2 = 2.0*(dxdt2 + dxdt)
        dxdt = self.dxdt()
        self.x = x + h6*(dxdt1 + dxdt + dxdt2)

    def store(self):
        self.his.append(self.x.mean(axis=0))
        self.var.append(self.x.var(axis=0))

        
class L40_ENKE(L40):
    """subclass for L40 with normal Kalman"""
    def assimilate(self, obs, var_obs, H, std_obs=None):
        if std_obs is None:
            std_obs = np.sqrt(var_obs)
            
        if H is None:
            H = np.eye(self.n)
            
        xb_bar = self.x.mean(axis=0) 
        xb_prime = self.x - xb_bar 
        Pb = np.sum((xb_prime[:, np.newaxis, :] * xb_prime[:, :, np.newaxis]), axis=0)/(members-1)
        R = var_obs * np.eye(H.shape[0])
        K = np.dot(np.dot(Pb,H.T),np.linalg.inv(np.dot(np.dot(H,Pb),H.T)+R))

        xa_bar = xb_bar + np.dot(K, (obs- np.dot(H,xb_bar)))
        xa_prime = np.zeros_like(xb_prime)
        for n in range(members):
            obs_prime = std_obs*np.random.randn(H.shape[0])
            xa_prime[n] = xb_prime[n] + np.dot(K, (obs_prime - np.dot(H,xb_prime[n])))
        self.x = xa_bar + xa_prime
        
class L40_ENSRF(L40):
    """subclass for L40 with modified Kalman"""
        
    def assimilate(self, obs, var_obs, std_obs=None, H= None):
        if std_obs is None:
            std_obs = np.sqrt(var_obs)

        if H is None:
            H = np.eye(self.n)
            
        for obsi, Hi in zip(obs,H):
            xb_bar = self.x.mean(axis=0) 
            xb_prime = self.x - xb_bar
            Pb = np.sum((xb_prime[:, np.newaxis, :] * xb_prime[:, :, np.newaxis]), axis=0)/(self.members-1)
            R = var_obs
            PbHT = np.dot(Pb, Hi.T)
            HPbHT = np.dot(Hi, PbHT)
            K = PbHT/(HPbHT+R)
            alpha = 1.0 /(1.0 + np.sqrt(R/(HPbHT + R)))       
            xa_bar = xb_bar + K*(obsi - np.dot(Hi,xb_bar))
            xa_prime = np.zeros_like(xb_prime)
            for n in range(members):
                xa_prime[n] = xb_prime[n] - alpha*(np.dot(K,(np.dot(Hi, xb_prime[n]))))

            self.x = xa_bar + xa_prime
        
def fourH(N=40):
# a function to make H as modes of variability, obs x grd
    x = np.arange(float(N))
    H = np.zeros((N, N))
    for mode in range(N):
        H[mode,:]=np.cos(mode * x * np.pi/float(N-1))
    return H



#Define and initialize truth    
truth = L40()
for _ in range(1000): # Initialize truth with a mature state
    truth.rk4step()

#build ensemble

ens = L40(members=members)
for n in range(members): 
    ens.x[n] = truth.x + 0.1 * np.random.randn(truth.n)

#Define assimilation 
members = 50
Nsteps = 1000
Nassim = 1
std_obs = 0.5
var_obs = std_obs**2
    
def run_group(Nassim, H=None, method):
    """ Nassim : Number of assimilations
        H : operator matrix
        method: 1 for traditional Kalman, 2 for enhanced Kalman   
    """

    if method==1:
        ens = L40_ENKE(ens)
    else:
        ens = L40_ENSRF(ens)
        
for n in range(Nsteps):
    #Step
    truth.rk4step()
    ens.rk4step()
    #Assimilate
    if np.mod(n, Nassim)==0:
        obs = truth.x[0] + std_obs*np.random.randn(truth.n)
        ens.assimilate(obs, var_obs)
    #Store
    truth.store()
    ens.store()
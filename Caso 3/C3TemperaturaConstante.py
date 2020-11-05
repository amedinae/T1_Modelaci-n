# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:34:07 2020

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def velocidad_reaccion(kf,kr,T):
    Ru = 8.315 #J/molK
    kf = kf[:,0]*T**kf[:,1]*np.exp(kf[:,2]/(Ru*T))
    kr = kr[:,0]*T**kr[:,1]*np.exp(kr[:,2]/(Ru*T))
    return kf,kr     

def dXdt(x,T,kf,kr):
    [[kf1,kf2,kf3,kf4,kf5,kf6],[kr1,kr2,kr3,kr4,kr5,kr6]] = velocidad_reaccion(kf,kr,T)
    return np.array([
                     -kf1*x[0]*x[1] + kr1*x[2]*x[3] + kf2*x[2]*x[4] - kr2*x[0]*x[3] + kf3*x[3]*x[4]-kr3*x[0]*x[5] - kf5*x[0]*x[6] + kr5*x[4]*x[1],
                     -kf1*x[0]*x[1] + kr1*x[2]*x[3] + kf5*x[0]*x[6] - kr5*x[4]*x[1] + kf6*x[2]*x[6]- kr6*x[3]*x[1],
                     kf1*x[0]*x[1] - kr1*x[2]*x[3] - kf2*x[2]*x[4] + kr2*x[0]*x[3] - kf4*x[2]*x[5] + kr4*(x[3]**2) - kf6*x[2]*x[6] + kr6*x[3]*x[1],
                     kf1*x[0]*x[1] - kr1*x[2]*x[3] +  kf2*x[2]*x[4] - kr2*x[0]*x[3] - kf3*x[3]*x[4] + kr3*x[0]*x[5] + 2*kf4*x[2]*x[5] - 2*kr4*(x[3]**2) + kf6*x[2]*x[6] - kr6*x[3]*x[1],
                     - kf2*x[2]*x[4] + kr2*x[0]*x[3] - kf3*x[3]*x[4] + kr3*x[0]*x[5] + kf5*x[0]*x[6] - kr5*x[4]*x[1],
                     kf3*x[3]*x[4]-kr3*x[0]*x[5] - kf4*x[2]*x[5] + kr4*(x[3]**2),
                     - kf5*x[0]*x[6] + kr5*x[4]*x[1] - kf6*x[2]*x[6] + kr6*x[3]*x[1]
                     ])
 
    
def euler(fi_in,dt):
    #fi_in = [x0,x1,x2,x3,x4] del tiempo conocido
    #fi_out = [x0,x1,x2,x3,x4] del tiempo futuro
    return fi_out + dXdt(fi_in)*dt

def kutta4(x,dt,T,kf,kr):
    k1 = dXdt(x,T,kf,kr)
    k2 = dXdt(x+1/2*k1*dt,T,kf,kr)
    k3 = dXdt(x+1/2*k2*dt,T,kf,kr)
    k4 = dXdt(x+k3*dt,T,kf,kr)
    f_out = x + dt/6*(k1+2*k2+2*k3+k4)
    return f_out
    
T = 1200.0 #K
dt = 1e-8 #s
tf = 35e-6#s
    
it = int(np.round(tf/dt))
    
t = np.linspace(0,tf,it+1)
x = np.zeros((it+1,7))
    
#filas son las iteraciones
#columnas son las especies

R= 0.08205746 # [atm*L/mol*K] 

xH2= 1.0 # moles
xO2= 0.3  # moles 
P= 1.2 # atm

x[0,1] = ((xO2/(xH2+xO2)*P)/(R*T))*10**(-3) #mol/cm^3
x[0,4] = ((xH2/(xH2+xO2)*P)/(R*T))*10**(-3) #mol/cm^3
    
kf = pd.read_csv('kf.dat', header=None, sep = " ",dtype=np.float64).values
kr = pd.read_csv('kr.dat', header=None, sep = " ",dtype=np.float64).values

kf[:,2] = -kf[:,2]*4.184
kr[:,2] = -kr[:,2]*4.184

for i in range(1,it+1):
    x[i,:] = kutta4(x[i-1,:],dt,T,kf,kr)
    

plt.plot(t,x[:,0],"-k",label="x_H")
plt.plot(t,x[:,1],"-b",label="x_O2")
plt.plot(t,x[:,2],"-g",label="x_O")
plt.plot(t,x[:,3],"-r",label="x_OH")
plt.plot(t,x[:,4],"-y",label="x_H2")
plt.plot(t,x[:,5],"-m",label="x_H20") 
plt.plot(t,x[:,6],"-c",label="x_H02")
plt.legend(loc="upper right")
plt.xlabel('tiempo(s)')
plt.ylabel('Concentración de las especies en mol/cm^3')
plt.grid()

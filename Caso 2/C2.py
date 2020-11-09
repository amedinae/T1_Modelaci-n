# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:07:19 2020

@author: Andres Medina
"""
import numpy as np
import matplotlib.pyplot as plt

def dQdt(x):
    A = 10;
    a = 1;
    Q1 = 0;
    g = -9.81;
    return np.array([x[1],
                     1/(2*x[0])*((A/a)**2-1)*(x[1])**2 - Q1/x[0]*(A/a)**2*x[1] + (A/a)**2*Q1**2/(2*x[0]) + A*g
                     ])

"""def dXdt(x):
    return np.array([Q3*c3-Q4/V*x[0],
                     Q5c5+Q4/V*x[0]+QA/V*x[3],
                     QB/V*x[1]-2*QC/V*x[2],
                     Q6*c6+QC/V*x[2]-QA/V/x[4]
                     ])"""

def kutta4(x):
    k1 = dQdt(x)
    k2 = dQdt(x+1/2*k1*dt)
    k3 = dQdt(x+1/2*k2*dt)
    k4 = dQdt(x+k3*dt)
    f_out = x + dt/6*(k1+2*k2+2*k3+k4)
    return f_out

dt = 1e-4 #s
tf = 10 #s
    
it = int(tf/dt)
    
t = np.linspace(0,tf,it+1)
x = np.zeros((it+1,2))
x[0] = [30,0]
for i in range(1,it+1):
    print(i)
    x[i,:] = kutta4(x[i-1,:])

#plt.plot(t,x[:,0],"-b",label="dVol/dt")    
plt.plot(t,x[:,1],"-b",label="Qx")
plt.legend(loc="upper right")
plt.xlabel('tiempo(s)')
plt.ylabel('Cambio de flujo volumetrico')
plt.grid()
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:34:49 2020

@author: Andres Medina
"""

import numpy as np
import matplotlib.pyplot as plt


def d1dt(x):
    Fd = 0.5*patm*Cd*Sb*vb**2 
    Fm =
    Wx =  
    return np.array([Q3*c3-Q4/V*x[0],
                     Q5*c5+Q4/V*x[0]+QA/V*x[3]-QB/V*x[1],
                     QB/V*x[1]-2*QC/V*x[2],
                     Q6*c6+QC/V*x[2]-QA/V*x[3]
                     ])

def d2dt(x,rhow,S,ma,mb):   
    return np.array([x[1],
                     1/(2*x[0])*((A/a)**2-1)*(x[1])**2 - Q1/x[0]*(A/a)**2*x[1] + (A/a)**2*Q1**2/(2*x[0]) + A*g
                     ])

def kutta4(x,sistema,Q2=0,Q3=0,Q5=0,Q6=0,V=0,c3=0,c5=0,c6=0,Q1=0,A=0,a=0,g=0):
    k1 = sistema(x,Q1,Q2,Q3,Q5,Q6,V,c3,c5,c6,A,a,g)
    k2 = sistema(x+1/2*k1*dt,Q1,Q2,Q3,Q5,Q6,V,c3,c5,c6,A,a,g)
    k3 = sistema(x+1/2*k2*dt,Q1,Q2,Q3,Q5,Q6,V,c3,c5,c6,A,a,g)
    k4 = sistema(x+k3*dt,Q1,Q2,Q3,Q5,Q6,V,c3,c5,c6,A,a,g)
    f_out = x + dt/6*(k1+2*k2+2*k3+k4)
    return f_out

dt = 1e-3 #s
tf = 40 #s
    
it = int(tf/dt)
    
t = np.linspace(0,tf,it+1)
x1 = np.zeros((it+1,2))
x2 = np.zeros((it+1,4))
x1[0] = [30,0]
x2[0] = [1,1,1,1]
A = 10;
a = 1;
g = -9.81;  
Q1 = 5
Q3 = 5
Q5 = 5
Q6 = 5
c3 = 0.01
c5 = 0.01
c6 = 0.01
V = 10
alpha =0.4
beta = 0.5
QD = 10

T0vacio = False
tvaciado = it

for i in range(1,it+1):
    if not T0vacio:
        x1[i,:] = kutta4(x1[i-1,:],dQdt,Q1=Q1,A=A,a=a,g=g)
    Q2 = Q1-x1[i-1,1]
    x2[i,:] = kutta4(x2[i-1,:],dXdt,Q2,Q3,Q5,Q6,V,c3,c5,c6)
    if x1[i,0]<1e-2 and not T0vacio:
        tvaciado = i
        T0vacio = True
    
plt.figure()
plt.plot(t[0:tvaciado],x1[0:tvaciado:,0],"-b",label="Vol")    
plt.plot(t[0:tvaciado],x1[0:tvaciado,1],"-g",label="dVol/dt")
plt.legend(loc="upper right")
plt.xlabel('tiempo(s)')
plt.ylabel('Cambio de flujo volumetrico')
plt.grid()

plt.figure()    
plt.plot(t,x2[:,0],"-b",label="x0")    
plt.plot(t,x2[:,1],"-g",label="x1")
plt.plot(t,x2[:,2],"-y",label="x2")    
plt.plot(t,x2[:,3],"-r",label="x3")
plt.legend(loc="upper right")
plt.xlabel('tiempo(s)')
plt.ylabel('Cantidad de soluto')
plt.grid()
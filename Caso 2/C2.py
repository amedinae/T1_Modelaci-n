# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:07:19 2020

@author: Andres Medina
"""
import numpy as np
import matplotlib.pyplot as plt

def dQdt(x,Q2=0,Q3=0,Q5=0,Q6=0,V=0,c3=0,c5=0,c6=0):
    A = 10;
    a = 1;
    Q1 = 0;
    g = -9.81;      
    return np.array([x[1],
                     1/(2*x[0])*((A/a)**2-1)*(x[1])**2 - Q1/x[0]*(A/a)**2*x[1] + (A/a)**2*Q1**2/(2*x[0]) + A*g
                     ])

def dXdt(x,Q2,Q3,Q5,Q6,V,c3,c5,c6):
    Q4 = Q2+Q3
    QA = Q4+Q5+2*Q6+(2-alpha-beta)*QD
    QB = 2*Q4+2*Q5+2*Q6+(2-alpha)*QD
    QC = Q4+Q5+Q6+QD
    return np.array([Q3*c3-Q4/V*x[0],
                     Q5*c5+Q4/V*x[0]+QA/V*x[3]-QB/V*x[1],
                     QB/V*x[1]-2*QC/V*x[2],
                     Q6*c6+QC/V*x[2]-QA/V*x[3]
                     ])

def kutta4(x,sistema,Q2=0,Q3=0,Q5=0,Q6=0,V=0,c3=0,c5=0,c6=0):
    k1 = sistema(x,Q2,Q3,Q5,Q6,V,c3,c5,c6)
    k2 = sistema(x+1/2*k1*dt,Q2,Q3,Q5,Q6,V,c3,c5,c6)
    k3 = sistema(x+1/2*k2*dt,Q2,Q3,Q5,Q6,V,c3,c5,c6)
    k4 = sistema(x+k3*dt,Q2,Q3,Q5,Q6,V,c3,c5,c6)
    f_out = x + dt/6*(k1+2*k2+2*k3+k4)
    return f_out

dt = 1e-5 #s
tf = 10 #s
    
it = int(tf/dt)
    
t = np.linspace(0,tf,it+1)
x1 = np.zeros((it+1,2))
x2 = np.zeros((it+1,4))
x1[0] = [30,0]
x2[0] = [1,1,1,1]
Q1 = 0
Q3 = 1
Q5 = 1
Q6 = 1
c3 = 0
c5 = 0
c6 = 0
V = 10
alpha =0.2
beta = 0.1
QD = 1
for i in range(1,it+1):
    x1[i,:] = kutta4(x1[i-1,:],dQdt)
    Q2 = Q1-x1[i-1,1]
    x2[i,:] = kutta4(x2[i-1,:],dXdt,Q2,Q3,Q5,Q6,V,c3,c5,c6)
    if x1[i,1]>1000 or x1[i,0]>1000:
        break
    
plt.figure()
plt.plot(t,x1[:,0],"-b",label="Vol")    
plt.plot(t,x1[:,1],"-g",label="Qx")
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
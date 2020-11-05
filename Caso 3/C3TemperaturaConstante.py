# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:34:07 2020

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt

def velocidad_reaccion(T):
    Ru = 8.315 #J/molK
    kf = np.zeros(6)
    kr = np.zeros(6)
    with open('kf.dat','r') as file1:
        for line in range(6):
            A,b,Ea = [float(x) for x in next(file1).split()]
            Ea = 4.184*Ea #J/mol
            kf[line] = A*T**b*np.exp(-Ea/(Ru*T))
            
    with open('kr.dat','r') as file2:
        for line in range(6):
            A,b,Ea = [float(x) for x in next(file2).split()]
            Ea = 4.184*Ea #J/mol
            kr[line] = A*T**b*np.exp(-Ea/(Ru*T))
    return kf,kr    

def funciones(t,x,T,i):
    #x --> [x0,x1,x2,x3,x4]
    [[kf1,kf2,kf3,kf4,kf5,kf6],[kr1,kr2,kr3,kr4,kr5,kr6]] = velocidad_reaccion(T)
    if(i==0):
        f = -kf1*x[0]*x[1] + kr1*x[2]*x[3] + kf2*x[2]*x[4] - kr2*x[0]*x[3] + kf3*x[3]*x[4]-kr3*x[0]*x[5] - kf5*x[0]*x[6] + kr5*x[4]*x[1]
    elif(i==1):
        f = -kf1*x[0]*x[1] + kr1*x[2]*x[3] + kf5*x[0]*x[6] - kr5*x[4]*x[1] + kf6*x[2]*x[6]- kr6*x[3]*x[1]
    elif(i==2):
        f = kf1*x[0]*x[1] - kr1*x[2]*x[3] - kf2*x[2]*x[4] + kr2*x[0]*x[3] - kf4*x[2]*x[5] + kr4*(x[3]**2) - kf6*x[2]*x[6] + kr6*x[3]*x[1]
    elif(i==3):
        f = kf1*x[0]*x[1] - kr1*x[2]*x[3] +  kf2*x[2]*x[4] - kr2*x[0]*x[3] - kf3*x[3]*x[4] + kr3*x[0]*x[5] + 2*kf4*x[2]*x[5] - 2*kr4*(x[3]**2) + kf6*x[2]*x[6] - kr6*x[3]*x[1] 
    elif(i==4):
        f = - kf2*x[2]*x[4] + kr2*x[0]*x[3] - kf3*x[3]*x[4] + kr3*x[0]*x[5] + kf5*x[0]*x[6] - kr5*x[4]*x[1]
    elif(i==5):
        f = kf3*x[3]*x[4]-kr3*x[0]*x[5] - kf4*x[2]*x[5] + kr4*(x[3]**2)
    elif(i==6):
        f = - kf5*x[0]*x[6] + kr5*x[4]*x[1] - kf6*x[2]*x[6] + kr6*x[3]*x[1]
    return f
 
def euler(fi_in,n,dt,t,T):
    #fi_in = [x0,x1,x2,x3,x4] del tiempo conocido
    #fi_out = [x0,x1,x2,x3,x4] del tiempo futuro
    #n es el número de variables
    #t es el instante de tiempo
    fi_out = np.zeros(n)
    for i in range(n):
        fi_out[i] = fi_in[i] + funciones(t,fi_in,T,i)*dt
    return fi_out

def kutta(x,n,dt,t,T):
    
    ## Paso 1
    k1 = np.zeros(n)
    for i in range(n):
        k1[i] = funciones(t,x,T,i)
    
    ## Paso 2
    k2 = np.zeros(n)
    xk2 = np.zeros(n)
    for i in range(n):
        xk2[i] = x[i] + (k1[i]/2)*dt
    for i in range(n):
        k2[i] = funciones(t+dt/2,xk2,T,i)
    
    ## Paso 3
    k3 = np.zeros(n)
    xk3 = np.zeros(n)    
    for i in range(n):
        xk3[i] = x[i] + (k2[i]/2)*dt
    for i in range(n):
        k3[i] = funciones(t+dt/2,xk3,T,i)    
    
    ## Paso 4
    k4 = np.zeros(n)
    xk4 = np.zeros(n)    
    for i in range(n):
        xk4[i] = x[i] + k3[i]*dt
    for i in range(n):
        k4[i] = funciones(t+dt,xk4,T,i) 
    
    ## Paso final
    xf = np.zeros(n)
    for i in range(n):
        xf[i]=x[i]+(dt/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
                          
    return xf
    
T = 1200.0 #K
dt = 1e-9 #s
tf = 30e-6#s
    
it = int(tf/dt)
    
t = np.zeros(it+1)
x = np.zeros((it+1,7))
    
#filas son las iteraciones
#columnas son las especies

R= 0.08205746 # [atm*L/mol*K] 

xH2= 1.0 # moles
xO2= 0.3  # moles 
P= 1.2 # atm

x[0,1] = ((xO2/(xH2+xO2)*P)/(R*T))*10**(-3) #mol/cm^3
x[0,4] = ((xH2/(xH2+xO2)*P)/(R*T))*10**(-3) #mol/cm^3
    

for i in range(1,it+1):
    t[i] = t[i-1] + dt
    x[i,:] = kutta(x[i-1,:],7,dt,t[i-1],T)
    

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

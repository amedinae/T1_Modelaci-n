# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:34:49 2020

@author: Andres Medina
"""

import numpy as np
import matplotlib.pyplot as plt


def d1dt(x,ma,mb,rhow,S,Sb,Cd,mu,theta,V,Vw0,rhoa0,patm):
    rho0 = 101325
    M = 0.0289654
    R = 8.31447
    T0 = 288.15
    L = 0.0065
    g = 9.81
    rhoatm = (rho0*M)/(R*T0)*(1-L*x[4]/T0)**((g*M)/(R*L)-1)
    Fd = 0.5*rhoatm*Cd*Sb*x[1]**2 
    Fm = mu*(ma+mb+x[0])*g*np.cos(theta)
    Wx = (ma+mb+x[0])*g*np.sin(theta)
    return np.array([-rhow*x[2]*S,
                     -(S**3*x[2]**3*rhow - 2*Fm*Sb**3 - 2*Sb**3*Wx - 2*Fd*Sb**3 + 2*Fd*Sb**2*x[1] + 2*Fm*Sb**2*x[1] + 2*Sb**2*x[1]*Wx - 2*x[3]*S*Sb**2*x[2] + 2*patm*S*Sb**2*x[2] + 2*S*Sb**3*x[2]*rhow - 2*S**2*Sb*x[2]**2*rhow + S*Sb**2*x[2]**3*rhow + 2*Fd*S*Sb*x[2] + 2*Fm*S*Sb*x[2] + 2*S*Sb*x[2]*Wx - 2*S**2*Sb**2*x[2]**2*rhow - 2*S*Sb**2*x[1]*x[2]**2*rhow + 2*S**2*Sb*x[1]*x[2]**2*rhow)/(2*Sb*(S*x[2]*ma - Sb**2*mb - Sb**2*x[0] - Sb**2*ma + S*x[2]*mb + Sb*x[1]*ma + Sb*x[1]*mb + Sb*x[1]*x[0])),
                     -(2*patm*Sb**2*x[2]*ma - 2*x[3]*Sb**2*x[2]*ma - 2*x[3]*Sb**2*x[2]*mb + 2*patm*Sb**2*x[2]*mb - 2*x[3]*Sb**2*x[2]*x[0] + 2*rhoatm*Sb**2*x[2]*x[0] - S**2*x[2]**3*ma*rhow - S**2*x[2]**3*mb*rhow + S**2*x[2]**3*x[0]*rhow + Sb**2*x[2]**3*ma*rhow + Sb**2*x[2]**3*mb*rhow + Sb**2*x[2]**3*x[0]*rhow + 2*Fd*Sb*x[2]*x[0] + 2*Fm*Sb*x[2]*x[0] + 2*Sb*x[2]*Wx*x[0] - 2*S*Sb*x[2]**2*x[0]*rhow + 2*Sb**2*x[1]*x[2]*ma*rhow + 2*Sb**2*x[1]*x[2]*mb*rhow + 2*Sb**2*x[1]*x[2]*x[0]*rhow - 2*Sb**2*x[1]*x[2]**2*ma*rhow - 2*Sb**2*x[1]*x[2]**2*mb*rhow - 2*Sb**2*x[1]*x[2]**2*x[0]*rhow)/(2*x[0]*(S*x[2]*ma - Sb**2*mb - Sb**2*x[0] - Sb**2*ma + S*x[2]*mb + Sb*x[1]*ma + Sb*x[1]*mb + Sb*x[1]*x[0])),
                     -( rhoa0*S)/( (rhoa0/rhow)*x[0] -rhoa0*V )*x[2]*x[3],
                     x[1]*np.sin(theta)
                     ])

def dM1dt(x,ma,mb,rhow,S,Sb,Cd,mu,theta,V,Vw0,rhoa0,patm):
    rho0 = 101325
    M = 0.0289654
    R = 8.31447
    T0 = 288.15
    L = 0.0065
    g = 9.81
    rhoatm = (rho0*M)/(R*T0)*(1-L*x[4]/T0)**((g*M)/(R*L)-1)
    Fd = 0.5*rhoatm*Cd*Sb*x[1]**2 
    Fm = mu*(ma+mb+x[0])*g*np.cos(theta)
    Wx = (ma+mb+x[0])*g*np.sin(theta)
    return np.array([-rhow*x[2]*S,
                     -1/(ma+mb)*(Fd+Fm+Wx)+1/(ma+mb)*(Sb*(x[3]-patm)-(Sb-S)**2/(2*Sb)*rhow*x[2]**2),
                     (Sb**2*(x[3]-patm))/(S)*(1/x[0]+1/(ma+mb)) - (x[2]**2*rhow)/(2*S)*((Sb**2 - S**2)/(x[0])+(Sb-S)**2/(mb+ma))-Sb/S*(ma+mb)*(Fd+Fm+Wx),
                     -( rhoa0*S)/( (rhoa0/rhow)*x[0] -rhoa0*V )*x[2]*x[3],
                     x[1]*np.sin(theta)
                     ])

#def d2dt(x,rhow,S,ma,mb):   
#    return np.array([x[1],
#                     1/(2*x[0])*((A/a)**2-1)*(x[1])**2 - Q1/x[0]*(A/a)**2*x[1] + (A/a)**2*Q1**2/(2*x[0]) + A*g
#                     ])

def kutta4(x,dt,ma,mb,rhow,S,Sb,Cd,mu,theta,V,Vw0,rhoa0,patm):
    k1 = dM1dt(x,ma,mb,rhow,S,Sb,Cd,mu,theta,V,Vw0,rhoa0,patm)
    k2 = dM1dt(x+1/2*k1*dt,ma,mb,rhow,S,Sb,Cd,mu,theta,V,Vw0,rhoa0,patm)
    k3 = dM1dt(x+1/2*k2*dt,ma,mb,rhow,S,Sb,Cd,mu,theta,V,Vw0,rhoa0,patm)
    k4 = dM1dt(x+k3*dt,ma,mb,rhow,S,Sb,Cd,mu,theta,V,Vw0,rhoa0,patm)
    f_out = x + dt/6*(k1+2*k2+2*k3+k4)
    return f_out

dt = 1e-6#s
tf = 1
    
it = int(tf/dt)
    
t = np.linspace(0,tf,it+1)
x1 = np.zeros((it+1,5))
x2 = np.zeros((it+1,4))
x1[0] = [10,0,0,2*101325,0]
x2[0] = [1,1,1,1]

ma = 0.001
mb = 0.1
rhow = 1
S = 0.01
Sb = 1
Cd = 0.1
mu = 0.1
theta = np.pi/2
V = 2e-3 # m**3
Vw0 = 1e-3 # m**3
rhoa0 = ma/(V-Vw0) #kg/m^3
patm = 101325

h = np.zeros((it+1,1))

for i in range(1,it+1):
    print(i)
    x1[i,:] = kutta4(x1[i-1,:],dt,ma,mb,rhow,S,Sb,Cd,mu,theta,V,Vw0,rhoa0,patm)
    if x1[i,0]<0:
        break

plt.figure()    
plt.plot(t,x1[:,0],"-b",label="x0")    
plt.legend(loc="upper right")
plt.xlabel('tiempo(s)')
plt.ylabel('Cantidad de soluto')
plt.grid()    
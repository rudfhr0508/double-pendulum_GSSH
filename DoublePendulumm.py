# -*- coding: utf-8 -*-
from __future__ import print_function   
from scipy.integrate import odeint 
import time
import math
import numpy as np
import pylab as py
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import pyplot as plt

m1 = 1
m2 = 1
L1 = 1
L2 = 1
g = 9.8

u0 = [np.pi, 0.1, np.pi, 0.1]

tfinal = 25.0
Nt = 751
t = np.linspace(0, tfinal, Nt)

def double_pendulum(u,t,m1,m2,L1,L2,g):
    du = np.zeros(4)
    c = np.cos(u[0]-u[2])
    s = np.sin(u[0]-u[2])
    du[0] = u[1]
    du[1] = ( m2*g*np.sin(u[2])*c - m2*s*(L1*c*u[1]**2 + L2*u[3]**2) - (m1+m2)*g*np.sin(u[0]) ) /( L1 *(m1+m2*s**2) )
    du[2] = u[3]
    du[3] = ((m1+m2)*(L1*u[1]**2*s - g*np.sin(u[2]) + g*np.sin(u[0])*c) + m2*L2*u[3]**2*s*c) / (L2 * (m1 + m2*s**2))
    return du

sol = odeint(double_pendulum, u0, t, args=(m1,m2,L1,L2,g))

u0 = sol[:,0]
u1 = sol[:,1]
u2 = sol[:,2]
u3 = sol[:,3]

x1 = L1*np.sin(u0)
y1 = -L1*np.cos(u0)
x2 = x1 + L2*np.sin(u2)
y2 = y1 - L2*np.cos(u2)

py.close('all')

py.figure(1)
py.plot(x1,y1,'.',color = '#0077BE',label = 'mass 1')
py.plot(x2,y2,'.',color = '#f66338',label = 'mass 2' )
py.legend()
py.xlabel('x (m)')
py.ylabel('y (m)')

fig = plt.figure()
ax = plt.axes(xlim=(-L1-L2-0.5, L1+L2+0.5), ylim=(-2.5, 1.5))
line1, = ax.plot([], [], 'o-',color = '#d2eeff',markersize = 12, markerfacecolor = '#0077BE',lw=2, markevery=10000, markeredgecolor = 'k')
line2, = ax.plot([], [], 'o-',color = '#ffebd8',markersize = 12, markerfacecolor = '#f66338',lw=2, markevery=10000, markeredgecolor = 'k')
line3, = ax.plot([], [], color='k', linestyle='-', linewidth=2)
line4, = ax.plot([], [], color='k', linestyle='-', linewidth=2)
line5, = ax.plot([], [], 'o', color='k', markersize = 10)
time_template = 'Time = %.1f s'
time_string = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    line5.set_data([], [])
    time_string.set_text('')
    return  line3,line4, line5, line1, line2, time_string

def animate(i):
    trail1 = 6
    trail2 = 8
    dt = t[2]-t[1]
    line1.set_data(x1[i:max(1,i-trail1):-1], y1[i:max(1,i-trail1):-1])
    line2.set_data(x2[i:max(1,i-trail2):-1], y2[i:max(1,i-trail2):-1])
    line3.set_data([x1[i], x2[i]], [y1[i], y2[i]])
    line4.set_data([x1[i], 0], [y1[i],0])
    line5.set_data([0, 0], [0, 0])
    time_string.set_text(time_template % (i*dt))
    return  line3, line4,line5,line1, line2, time_string

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=Nt, interval=1000*(t[2]-t[1])*0.8, blit=True)

anim.save('double_pendulum_animation.gif', fps=1.0/(t[2]-t[1]), writer = 'pillow')

plt.show()

u0 = np.array([np.pi, 0.1, np.pi, 0.1])
delta0 = 1e-8
u0_perturbed = u0 + np.array([delta0, 0, 0, 0])

u_traj1 = odeint(double_pendulum, u0, t, args=(m1,m2,L1,L2,g))
u_traj2 = odeint(double_pendulum, u0_perturbed, t, args=(m1,m2,L1,L2,g))

d0 = np.linalg.norm(u_traj1[0]-u_traj2[0])
log_diffs = []  
t_renorm = []  
renorm_interval = 1.0
current_index = 0

for i in range(1, len(t)):
    d = np.linalg.norm(u_traj1[i]-u_traj2[i])
    if t[i] - t[current_index] >= renorm_interval:
        log_diffs.append(np.log(d/d0))
        t_renorm.append(t[i])
        direction = (u_traj2[i]-u_traj1[i]) / d
        u_traj2[i] = u_traj1[i] + d0 * direction
        current_index = i

lambda_estimate = np.mean(log_diffs) / renorm_interval
print("Estimated Lyapunov exponent:", lambda_estimate)
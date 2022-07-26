from scipy.optimize import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sympy as sp
import numpy as np
from numpy import *
from matplotlib.pyplot import *
from sympy import symbols, solve, Eq
import time
b = 100
def rosen(x):
    return (100*((x[1] - x[0]**2)**2) + (1 - x[0])**2)
k = 0
f = lambda x,y: (x-1)**2 + b*(y-x**2)**2

def plot3d():
    # Initialize figure 
    figRos = plt.figure(figsize=(12, 7))
    axRos = figRos.gca(projection='3d')

    # Evaluate function
    X = np.arange(-2, 2, 0.15)
    Y = np.arange(-1, 3, 0.15)
    X, Y = np.meshgrid(X, Y)
    Z = f(X,Y)

    # Plot the surface
    surf = axRos.plot_surface(X, Y, Z, cmap=cm.gist_heat_r,
                        linewidth=0, antialiased=False)
    axRos.set_zlim(0, 2000)
    figRos.colorbar(surf, shrink=0.5, aspect=10) 
    plot2d(X, Y, Z)

def plot2d(X, Y, Z):
    df = lambda x,y: np.array([2*(x-1) - 4*b*(y - x**2)*x, \
                         2*b*(y-x**2)])
    F = lambda X: f(X[0],X[1])
    dF = lambda X: df(X[0],X[1])
    x0 = np.array([-1.4,1.1])
    print(F(x0))
    print(dF(x0))
    # Initialize figure 
    plt.figure(figsize=(12, 7))
    plt.contour(X,Y,Z,200)
    plt.plot([x0[0]],[x0[1]],marker='o',markersize=15, color ='r')

def callback(x):
    global k
    if k == 0:
        global xant, yant
        xant = x[0]
        yant = x[1] 
        k = 1
        xpoints = np.array([x[0], -1.4])
        ypoints = np.array([x[1], 1.1])
        plot3d()
        plt.plot(xpoints, ypoints)
    xpoints = np.array([x[0], xant])
    ypoints = np.array([x[1], yant])
    
    plt.plot(xpoints, ypoints)

    xant = x[0]
    yant = x[1] 
    k = 1

def constraint(x):
    return (((x[0]**2)/9 + (x[1]**2)/16))

cons = ({'type': 'eq', 'fun':constraint})
r = minimize(rosen, (-1.4, 1.1), method = 'CG', callback=callback, options={'return_all' : True}, constraints=cons)
print(r)
t = linspace(0,360,360)
el_x = 3*cos(radians(t)) 
el_y = 4*sin(radians(t))
plt.plot(el_x, el_y)
# plot3d()

plt.show()
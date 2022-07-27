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
x_0= -1.4
y_0 = 1.1

def rosen(x):
    return (100*((x[1] - x[0]**2)**2) + (1 - x[0])**2)
k = 0
f = lambda x,y: (x-1)**2 + b*(y-x**2)**2

def plot3d():
    # Initialize figure 
    figRos = plt.figure(figsize=(12, 7))
    axRos = figRos.gca(projection='3d')

    # Evaluate function
    X = np.arange(-3, 3, 0.15)
    Y = np.arange(-4, 4, 0.15)
    X, Y = np.meshgrid(X, Y)
    Z = f(X,Y)

    # Plot the surface
    surf = axRos.plot_surface(X, Y, Z, cmap=cm.gist_heat_r,
                        linewidth=0, antialiased=False)
    axRos.set_zlim(0, 4000)
    figRos.colorbar(surf, shrink=0.5, aspect=10) 
    # barreira
    t = linspace(0,360,360)
    el_x = 3*cos(radians(t)) 
    el_y = 4*sin(radians(t))
    el_z = np.linspace(0, 4000, 50)
    t, el_z = np.meshgrid(t, el_z)
    axRos.plot_surface(el_x, el_y, el_z, cmap='plasma')
    plot2d(X, Y, Z)

def plot2d(X, Y, Z):
    df = lambda x,y: np.array([2*(x-1) - 4*b*(y - x**2)*x, \
                         2*b*(y-x**2)])
    F = lambda X: f(X[0],X[1])
    dF = lambda X: df(X[0],X[1])
    x0 = np.array([x_0, y_0])

    # Initialize figure 
    plt.figure(figsize=(12, 7))
    plt.contour(X,Y,Z,200)
    plt.plot([x0[0]],[x0[1]],marker='o',markersize=15, color ='r')
    # ellipse
    t = linspace(0,360,360)
    el_x = 3*cos(radians(t)) 
    el_y = 4*sin(radians(t))
    plt.plot(el_x, el_y)    

def callback(x):
    global k
    if k == 0:
        global xant, yant
        xant = x[0]
        yant = x[1] 
        k = 1
        xpoints = np.array([x[0], x_0])
        ypoints = np.array([x[1], y_0])
        plot3d()
        plt.plot(xpoints, ypoints)
    xpoints = np.array([x[0], xant])
    ypoints = np.array([x[1], yant])
    
    plt.plot(xpoints, ypoints)

    xant = x[0]
    yant = x[1] 
    k = 1
def constraint2(x, y):
    return (((x**2)/9 + (y**2)/16))

def constraint(x):
    return (((x[0]**2)/9 + (x[1]**2)/16))

cons = ({'type': 'ineq', 'fun':constraint})
r = minimize(rosen, (x_0, y_0), method = 'CG', callback=callback, options={'return_all' : True}, constraints=cons)
print(r)

plt.show()

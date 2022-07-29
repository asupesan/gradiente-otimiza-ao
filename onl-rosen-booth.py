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
k = 0
def rosen(x):
    return (100*((x[1] - x[0]**2)**2) + (1 - x[0])**2)

f = lambda x,y: (1-x)**2 + b*(y-x**2)**2

def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
f2 = lambda x,y: (x + 2*y - 7)**2 + (2*x + y - 5)**2
constraint2LAMBDA = lambda x,y: y - (x - 2)**3 - x**2 - 10
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
                        linewidth=0, antialiased=False, alpha=.4)
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

#booth
def plot3d2():
    # Initialize figure 
    figRos = plt.figure(figsize=(12, 7))
    axRos = figRos.gca(projection='3d')

    # Evaluate function
    X = np.arange(-5, 5, 0.15)
    Y = np.arange(-5, 5, 0.15)
    X, Y = np.meshgrid(X, Y)
    Z = f2(X,Y)

    # Plot the surface
    surf = axRos.plot_surface(X, Y, Z, cmap=cm.gist_heat_r,
                        linewidth=0, antialiased=False, alpha=.4)
    axRos.set_zlim(0, 1000)
    figRos.colorbar(surf, shrink=0.5, aspect=10) 
    # barreira

    el_z = constraint2LAMBDA(X,Y)
    axRos.plot_surface(X, Y, el_z, cmap='plasma')
    X = np.arange(-5, 5, 0.15)
    Y = np.arange(-5, 41, 0.15)
    X, Y = np.meshgrid(X, Y)
    Z = f2(X,Y)
    plot2d2(X, Y, Z)

def plot2d2(X, Y, Z):
    df = lambda x,y: np.array([2*(x-1) - 4*b*(y - x**2)*x, \
                         2*b*(y-x**2)])
    F = lambda X: f(X[0],X[1])
    dF = lambda X: df(X[0],X[1])
    x0 = np.array([x_0, y_0])

    # Initialize figure 
    plt.figure(figsize=(12, 7))
    plt.contour(X,Y,Z,200)
    plt.plot([x0[0]],[x0[1]],marker='o',markersize=15, color ='r')
    # FALTA ARRUMA O PLOT DA RESTRIÇAO 2D
    
    el_x = linspace(0, 0) 
    el_y = linspace(2, 41)
    plt.plot(el_x, el_y)    
# booth end
def callback(x):
    global k
    if k == 0:
        global xant, yant
        xant = x[0]
        yant = x[1] 
        k = 1
        xpoints = np.array([x[0], x_0])
        ypoints = np.array([x[1], y_0])
        if inp == 'booth':
            plot3d2()
        else:
            plot3d()
        plt.plot(xpoints, ypoints)
    xpoints = np.array([x[0], xant])
    ypoints = np.array([x[1], yant])
    
    plt.plot(xpoints, ypoints)

    xant = x[0]
    yant = x[1] 
    k = 1
def constraint2(x):
    return x[1] - (x[0] - 2)**3 - x[0]**2 - 10

def constraint(x):
    return (((x[0]**2)/9 + (x[1]**2)/16))

cons = ({'type': 'ineq', 'fun':constraint})
cons2 = ({'type': 'ineq', 'fun':constraint2})
global inp
inp = input("booth ou rosen: ")
if inp == 'booth':
    r = minimize(booth, (x_0, y_0), method = 'SLSQP', callback=callback, options={'return_all' : True}, constraints=cons2)
if inp == 'rosen':
    r = minimize(rosen, (x_0, y_0), method = 'SLSQP', callback=callback, options={'return_all' : True}, constraints=cons)
print(r)

plt.show()

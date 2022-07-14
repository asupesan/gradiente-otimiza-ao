import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sympy as sp
import numpy as np
from numpy import *
from matplotlib.pyplot import *
from sympy import symbols, solve, Eq

x, y = sp.symbols('x y')
def gradient(f, g, x0, y0):
    i=1
    c=1
    xant = x0
    yant = y0

    xlist = np.linspace(0, 5, 13)
    ylist = np.linspace(0, 5, 13)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt((X-3)**2 + (Y-2)**2)
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Curvas de nivel')
    
    while c != 0:
        print("-----------------------------", i,"-----------------------------")
        u = sp.Symbol('u')
        #f = f + u*1/g
        #print(f)
        dfdx = sp.diff(f, x)
        dfdy = sp.diff(f, y)
        


        dfdx0 = dfdx.subs([(x, x0), (y, y0)])
        dfdy0 = dfdy.subs([(x, x0), (y, y0)])
        # print(dfdx0)
        # print(dfdy0)

        a = sp.Symbol('a')
        x1 = x0 - a*dfdx0
        y1 = y0 - a*dfdy0
        #print(x1)
        #print(y1)
        #print(type(x1))
        fa = f.subs([(x, x1), (y, y1)])
        #print(fa)
        dfda = sp.diff(fa, a)
        #print(dfda)

        eq = Eq(dfda, 0)
        a = solve(eq)
        #print(type(a))
        
        x0 = x0 - a[0]*dfdx0
        y0 = y0 - a[0]*dfdy0

        xpoints = np.array([x0, xant])
        ypoints = np.array([y0, yant])
       
        plt.plot(xpoints, ypoints)
        
        print("x = ", float(x0))
        print("y = ", float(y0))
        i += 1
        #print(xmin)
        #print(x0)
        dfdx0 = dfdx.subs([(x, float(x0)), (y, float(y0))])
        dfdy0 = dfdy.subs([(x, float(x0)), (y, float(y0))])
        if dfdx0 == 0 and dfdy0 == 0:
            c = 0
        #print(float(xant))
        #print(float(x0))
        xant = x0
        yant = y0
        
f = (((x-3)**2)/4 + ((y-2)**2)/9) + 13
g = (x**2)/9 + (y**2)/16
print(type(f))
x0 = 4
y0 = 4

gradient(f, g, x0, y0)

t = linspace(0,90,90)
el_x = 3*cos(radians(t)) 
el_y = 4*sin(radians(t))
plt.plot(el_x, el_y) 
plt.show()

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sympy as sp
import numpy as np
from numpy import *
from matplotlib.pyplot import *
from sympy import symbols, solve, Eq
import time
x, y = sp.symbols('x y')

def gradient(f, g, x0, y0):
    start_time = time.time()
    
    i=1
    c=1
    xant = x0
    yant = y0

    # xlist = np.linspace(0, 5, 13)
    # ylist = np.linspace(0, 5, 13)
    # X, Y = np.meshgrid(xlist, ylist)
    # Z = np.sqrt((X-3)**2 + (Y-2)**2)
    # fig,ax=plt.subplots(1,1)
    # cp = ax.contour(X, Y, Z)
    # plt.clabel(cp, inline=True, fontsize=8)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Curvas de nivel')
    
    while c != 0:
        print("-----------------------------", i,"-----------------------------")
        #u = sp.Symbol('u')
        u = 1
        #f = f - u/g
        f = f + u*g**2
        print(f)
        dfdx = sp.diff(f, x)
        dfdy = sp.diff(f, y)
        
        dfdx0 = dfdx.subs([(x, x0), (y, y0)])
        dfdy0 = dfdy.subs([(x, x0), (y, y0)])

        a = sp.Symbol('a')
        x1 = x0 - a*dfdx0
        y1 = y0 - a*dfdy0

        fa = f.subs([(x, x1), (y, y1)])

        dfda = sp.diff(fa, a)

        eq = Eq(dfda, 0)
        a = solve(eq)
        
        x0 = x0 - a[0]*dfdx0
        y0 = y0 - a[0]*dfdy0

        xpoints = np.array([x0, xant])
        ypoints = np.array([y0, yant])
  
        plt.plot(xpoints, ypoints)
        
        print("x = ", float(x0))
        print("y = ", float(y0))
        i += 1

        dfdx0 = dfdx.subs([(x, float(x0)), (y, float(y0))])
        dfdy0 = dfdy.subs([(x, float(x0)), (y, float(y0))])
        # # ex2- start
        # g0 = g.subs([(x, float(x0)), (y, float(y0))])
        # print(g0)
                                                            
        # if g0 > 0 :
        #     f = f + g**2
        #     print("pudim")
        # # ex2- end
        if dfdx0 == 0 and dfdy0 == 0:
            c = 0
        if i > 3:
            if x0/xant < 1.1 and x0/xant > 0.9 and y0/yant < 1.1 and y0/yant > 0.9:
                c = 0
        xant = x0
        yant = y0
        print("--- %s seconds ---" % (time.time() - start_time))

f = (100*(y - x**2)**2 + (1 - x)**2)   #ex1
f = (x + 2*y - 7)**2 + (2*x + y - 5)**2 #ex2
#f = (((x-3)**2)/4 + ((y-2)**2)/9) + 13

g = (((x**2)/9 + (y**2)/16))#ex 1
g = (y - (x - 2)**3 - (x**2) - 10) #ex 2

x0 = 1.1
y0 = 1.1

gradient(f, g, x0, y0)

t = linspace(0,90,90)
el_x = 3*cos(radians(t)) 
el_y = 4*sin(radians(t))
plt.plot(el_x, el_y) 
plt.show()

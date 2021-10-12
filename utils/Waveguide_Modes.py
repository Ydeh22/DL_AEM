import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from numpy import pi,sqrt,sin,cos,gradient,abs,arctan,exp,tan
from Drude_Model import drude_model_Si, drude_lorentz_model_Si

from scipy.optimize import fsolve, root
from scipy.special import j1,k1,jvp,kvp
from scipy.special import hankel1

def hybrid_mode_EH_graphical(r_0,h_0,log_n,f0_0):

    f0 = f0_0 * 1e12
    r = r_0 * 1e-6
    h = h_0 * 1e-6

    epsilon_0 = 1
    epsilon_2 = epsilon_0

    e1,e2 = drude_model_Si(log_n, f0)
    # e1,e2 = drude_lorentz_model_Si(11.68,0,0,f0)
    epsilon_1 = e1

    c = 2.99792458e8
    lam = c / f0
    k0 = 2 * pi / lam
    n = 1
    m = 0

    alpha = 3.8317 / r
    beta = sqrt(k0**2 * epsilon_1 - alpha**2)
    p = sqrt(alpha**2 - k0**2 * epsilon_0)
    f_res_1 = tan(beta * h / 2 - m * pi / 2)
    f_res_2 = (p / beta)

    return f_res_1, f_res_2

def hybrid_mode_HE_graphical(r_0,h_0,log_n,f0_0):

    f0 = f0_0 * 1e12
    r = r_0 * 1e-6
    h = h_0 * 1e-6

    epsilon_0 = 1
    epsilon_2 = epsilon_0

    e1,e2 = drude_model_Si(log_n, f0)
    # e1,e2 = drude_lorentz_model_Si(11.68,0,0,f0)
    epsilon_1 = e1

    c = 2.99792458e8
    lam = c / f0
    k0 = 2 * pi / lam
    n = 1
    m = 0

    beta = pi/h
    alpha = sqrt(k0**2 * epsilon_1 - beta**2)
    gamma1 = alpha
    gamma2 = sqrt(beta**2 - k0**2 * epsilon_0)
    u = gamma1 * r
    v = gamma2 * r
    # eta1 = jvp(1,u)/(u*j1(u)) + kvp(1,v)/(v*k1(v))
    # eta2 = (epsilon_1*k0**2)*jvp(1, u) / (u * j1(u)) + (k0**2)*kvp(1, v) / (v * k1(v))

    eta1 = jvp(n - 1, u) / (u * jvp(n, u)) - n / u ** 2
    eta2 = 1j * hankel1(n - 1, 1, 1j * v) / v / hankel1(n, 1, 1j * v) - n / v ** 2
    f_res_1 = abs((eta1 + eta2) * (k0 ** 2 * epsilon_1 * eta1 + k0 ** 2 * epsilon_2 * eta2))
    f_res_2 = (n ** 2 * beta ** 2 * (1 / u ** 2 + 1 / v ** 2) ** 2)

    # eta1 = jvp(n-1,u)/(u*jvp(n,u)) + hankel1(n-1,v)/(v*hankel1(n,v))
    # eta2 = (epsilon_1*k0**2)*jvp(n-1, u) / (u * jvp(n,u)) + (k0**2)*hankel1(n-1, v) / (v * hankel1(n,v))
    # eta3 = (beta**2) * (1/u**2 + 1/v**2)**2
    #
    # f_res_1 = abs(eta1*eta2)
    # f_res_2 = eta3

    return f_res_1, f_res_2


def hybrid_mode(r,h,log_n,mode):

    if mode == 'EH':

        def hybrid_mode_EH_solve(f0):

            y1,y2 = hybrid_mode_EH_graphical(r,h,log_n,f0)
            return y1 - y2

        f_res = fsolve(hybrid_mode_EH_solve, [45])

        return f_res

    if mode == 'HE':

        def hybrid_mode_HE_solve(f0):
            y1, y2 = hybrid_mode_HE_graphical(r, h, log_n, f0)
            return y1 - y2

        f_res = fsolve(hybrid_mode_HE_solve, [20])
        return f_res

if __name__ == "__main__":

    # f0 = hybrid_mode(1.5,1.5,35,'HE')
    # print(f0)

    for i in range(21):
        r = 1.3 + 0.1*i
        f0 = hybrid_mode(r, 1.6, 41, 'EH')
        print(f0[0])

    # res = 1000
    # f = np.linspace(20,40,res)
    #
    # fs = np.empty((res,2))
    #
    # for ind,w in enumerate(f):
    #     y1,y2 = hybrid_mode_HE_graphical(2,1,40,w)
    #     fs[ind,0] = y1
    #     fs[ind,1] = y2
    #
    # plt.plot(f,fs[:,0],f,fs[:,1])
    # plt.show()


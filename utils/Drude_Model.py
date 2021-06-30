import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from numpy import pi,sqrt,sin,cos,gradient,abs,arctan,exp

# Define some universal constants
c=3e8
e0=(10**7)/(4*pi*c**2)
m0=4*pi*10**(-7)
z0=sqrt(m0/e0)
q=1.602e-19
m_e = 9.109e-31


def drude_lorentz_model_Si(eps_inf, wp, g, f0):
    w = 2 * pi * f0
    epsilon = eps_inf - wp**2 / (w**2 + 1j * w * g)
    e1 = epsilon.real
    e2 = epsilon.imag
    return e1,e2

def drude_model_Si(log_n,f):
    # Define some universal constants
    c = 3e8
    e0 = (10 ** 7) / (4 * pi * c ** 2)
    m0 = 4 * pi * 10 ** (-7)
    z0 = sqrt(m0 / e0)
    q = 1.602e-19
    m_e = 9.109e-31

    # Specify some material parameters
    eps_inf = 11.68
    m_eff = 0.27 * m_e
    n_d = exp(log_n)

    # Specify scattering rate
    # scat = 1e13
    # tau = 1/scat
    # mob = q*tau/m_eff

    # Specify mobility
    # mob = 1200
    mu_0 = 65
    mu_1 = 1265
    alpha = 0.72
    n_ref = 8.5e16
    mob = mu_0 + mu_1 / (1 + (n_d / n_ref) ** alpha)
    scat = (1e4) * q / (mob * m_eff)
    tau = 1 / scat

    # Specify the frequency range in hertz and radians, the free space
    # wavevector and wavelength
    f0 = f*1e12
    w = 2 * pi * f0
    k0 = w / c
    lambda0 = 2 * pi / k0

    # Drude model
    s_DC = (n_d * tau * q ** 2) / m_eff
    wp = sqrt((n_d * 1e6 * q ** 2) / (e0 * m_eff))
    s = s_DC / (1 - 1j * w * tau)
    s1 = s.real
    s2 = s.imag

    # Drude-Lorentz model
    er = eps_inf + wp ** 2 / (-w ** 2 - 1j * scat * w)
    eps = e0 * er  # permittivity
    e1 = er.real
    e2 = er.imag

    return e1,e2
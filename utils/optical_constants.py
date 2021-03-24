import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from math import pi
from numpy import sqrt as root
from torch import pow, add, mul, div, sqrt, square, cos, sin, conj, exp, abs

def matrix_method_slab(er, mr, d, f):

    cuda = True if torch.cuda.is_available() else False

    # Fundamental constants
    c = 3e8
    e0 = (10 ** 7) / (4 * pi * c ** 2)
    m0 = 4 * pi * 10 ** (-7)
    z0 = root(m0 / e0)

    w_numpy = 2 * pi * f
    k0 = w_numpy / c

    # if cuda:
    #     w = torch.tensor(w_numpy).cuda()
    #     k0 = torch.tensor(k0).cuda()
    # else:
    #     w = torch.tensor(w_numpy)
    #     k0 = torch.tensor(k0)
    w = w_numpy
    # mr = torch.ones_like(er)
    # if cuda:
    #     mr = mr.cuda()

    eps = e0*er
    mu = m0*mr
    n = sqrt(mul(mr, er))
    k = div(mul(w, n), c)
    # z = sqrt(div(mu, eps))

    # M = cos(mul(k,d)) - 0.5*1j*mul((div(k,k0)+div(k0,k)),(sin(mul(k,d))))
    M12_TE = cos(mul(k, d)) + 0.5*1j*mul((mul(div(1, mr), div(k, k0)) + mul(mr, div(k0, k))), (sin(mul(k, d))))
    M22_TE = cos(mul(k, d)) - 0.5*1j*mul((mul(div(1, mr), div(k, k0)) + mul(mr, div(k0, k))), (sin(mul(k, d))))
    r = div(M12_TE,M22_TE)
    t = div(1, M22_TE)
    # T = (mul(t, conj(t)).real).float()
    # R = (mul(r, conj(r)).real).float()
    return r, t

def lorentzian(w, w0, wp , g, eps_inf=0):
    num1 = mul(square(wp), add(square(w0), -square(w)))
    num2 = mul(square(wp), mul(w, g))
    denom = add(square(add(square(w0), -square(w))), mul(square(w), square(g)))
    e1 = div(num1, denom)
    e2 = div(num2, denom)
    e1 += eps_inf
    return e1,e2
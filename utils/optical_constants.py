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

    d = d * 1e-6
    w_numpy = 2 * pi * f * 1e12
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

    j = torch.tensor([0 + 1j], dtype=torch.cfloat).expand_as(er)
    if cuda:
        j = j.cuda()

    eps = e0*er
    mu = m0*mr
    e1 = er.real
    e2 = F.relu(er.imag)
    er = add(e1, mul(e2, j))
    mu1 = mr.real
    mu2 = F.relu(mr.imag)
    mr = add(mu1, mul(mu2, j))
    n = sqrt(mul(mr, er))
    n1 = n.real
    n2 = F.relu(n.imag)
    n = add(n1, mul(n2, j))
    k = div(mul(w, n), c)
    k1 = k.real
    k2 = F.relu(k.imag)
    k = add(k1, mul(k2, j))
    z = sqrt(div(mu, eps))

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

class matrix_method_slab_debug:

    def __init__(self, er, mr, d, f, cuda=False):

        self.opt_const(er, mr, d, f, cuda)

    def opt_const(self, er, mr, d, f, cuda=False):

        # Fundamental constants
        c = 3e8
        e0 = (10 ** 7) / (4 * pi * c ** 2)
        m0 = 4 * pi * 10 ** (-7)
        z0 = root(m0 / e0)

        d = d * 1e-6
        w = 2 * pi * f * 1e12
        k0 = w / c

        j = torch.tensor([0 + 1j], dtype=torch.cfloat).expand_as(er)
        if cuda:
            j = j.cuda()

        eps = e0*er
        mu = m0*mr
        e1 = er.real
        e2 = F.relu(er.imag)
        er = add(e1, mul(e2, j))
        mu1 = mr.real
        mu2 = F.relu(mr.imag)
        mr = add(mu1, mul(mu2, j))
        n = sqrt(mul(mr, er))
        n1 = n.real
        n2 = F.relu(n.imag)
        n = add(n1, mul(n2, j))
        k = div(mul(w, n), c)
        # k1 = k.real
        # k2 = F.relu(k.imag)
        # k = add(k1, mul(k2, j))
        z = sqrt(div(er, mr))
        M12_TE = 0.5 * 1j * mul((z - div(1,z)), (sin(mul(k, d))))
        M22_TE = cos(mul(k, d)) - 0.5 * 1j * mul((z + div(1,z)), (sin(mul(k, d))))
        # M12_TE = 0.5*1j*mul((mul(div(1, mu), div(k, k0)) + mul(mu, div(k0, k))), (sin(mul(k, d))))
        # M22_TE = cos(mul(k, d)) - 0.5*1j*mul((mul(div(1, mu), div(k, k0)) + mul(mu, div(k0, k))), (sin(mul(k, d))))
        r = div(M12_TE,M22_TE)
        t = div(1, M22_TE)
        # T = (mul(t, conj(t)).real).float()
        # R = (mul(r, conj(r)).real).float()

        self.w = f
        self.k0 = k0
        self.k = k
        self.n = n
        self.z = z
        self.eps = er
        self.mu = mr
        self.M12 = M12_TE
        self.M22 = M22_TE
        self.r = r
        self.t = t
        self.T = square(abs(t))
        self.R = square(abs(r))

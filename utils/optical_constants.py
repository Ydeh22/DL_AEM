import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from math import pi
from numpy import sqrt as root
from torch import pow, add, mul, div, sqrt, square, \
                     cos, sin, conj, exp, abs, arctan, tan
from torch import square as sq
from utils.custom_functions import real_check, imag_check

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

    eps = mul(e0,er)
    mu = mul(m0,mr)
    # e1 = er.real
    # e2 = F.relu(er.imag)
    # er = add(e1, mul(e2, j))
    n = sqrt(mul(mr, er))
    n = imag_check.apply(n)
    # k = div(mul(w, n), c)
    # z = sqrt(div(mu, eps + 1e-5))

    # Spatial dispersion
    theta = mul(w * d, sqrt(mul(eps, mu))).type(torch.cfloat)
    magic = div(tan(0.5 * theta), 0.5 * theta).type(torch.cfloat)
    eps_av = mul(magic, er)
    mu_av = mul(magic, mr)
    n_av = sqrt(mul(mu_av, eps_av))
    n = imag_check.apply(n)

    # eps_av = er
    # mu_av = mr
    # n_av = n
    k = div(mul(w, n_av), c)
    z = sqrt(div(eps_av, mu_av))

    # R2 = sq(abs(div((mu_av - n_av), (mu_av + n_av))))
    r = div((mu_av - n_av), (mu_av + n_av))
    # phiR = arctan(div(r2.imag, r2.real))
    # phiR2 = arctan(div(2 * n_av.imag, (1 - sq(n_av.real) - sq(n_av.imag))))
    # T2 = exp(-2 * mul(k.imag, d)) * sq((div(n_av, mu_av)).real * sq(abs(div(2 * mu_av, (n_av + mu_av)))))
    # T3 = mul(exp(-2 * mul(k.imag, d)), sq(
    #     (div(mul(n_av.real, mu.real)
    #          + mul(n_av.imag, mu.imag), (sq(mu.real) +
    #                                      sq(mu.imag)))) * m0 * sq(abs(div(2 * mu_av, (n_av + mu_av))))))
    t = exp(-1 * mul(k.imag, d)) * div(2 * mr, (n + mr)) * sqrt(div(n, mr))
    # phiT2 = arctan(div(t2.imag, t2.real))

    # M12_TE = cos(mul(k, d)) + 0.5*1j*mul((mul(div(1, mr + 1e-5), div(k, k0)) + mul(mr, div(k0, k))), (sin(mul(k, d))))
    # M22_TE = cos(mul(k, d)) - 0.5*1j*mul((mul(div(1, mr + 1e-5), div(k, k0)) + mul(mr, div(k0, k))), (sin(mul(k, d))))

    # M12_TE = 0.5 * 1j * mul((z - div(1, z + 1e-5)), (sin(mul(k, d))))
    # M22_TE = cos(mul(k, d)) - 0.5 * 1j * mul((z + div(1, z + 1e-5)), (sin(mul(k, d))))
    #
    # r = div(M12_TE,M22_TE + 1e-5)
    # t = div(1, M22_TE + 1e-5)

    # T = (mul(t, conj(t)).real).float()
    # R = (mul(r, conj(r)).real).float()
    return r,t

def lorentzian(w, w0, wp , g, eps_inf=0):
    num1 = mul(sq(wp), add(sq(w0), -sq(w)))
    num2 = mul(sq(wp), mul(w, g))
    denom = add(sq(add(sq(w0), -sq(w))), mul(sq(w), sq(g)))
    e1 = div(num1, denom + 1e-5)
    e2 = div(num2, denom + 1e-5)
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
        p = 6 * 1e-6
        d = d * 1e-6
        w = 2 * pi * f * 1e12
        k0 = w / c

        j = torch.tensor([0 + 1j], dtype=torch.cfloat).expand_as(er)
        if cuda:
            j = j.cuda()

        eps = mul(e0, er)
        mu = mul(m0, mr)
        n = sqrt(mul(mr, er))
        # n1 = n.real
        # n2 = F.relu(n.imag)
        # n = add(n1, mul(n2, j))

        # # Spatial dispersion
        theta = mul(w*d, sqrt(mul(eps,mu))).type(torch.cfloat)
        magic = div(tan(0.5*theta),0.5*theta).type(torch.cfloat)
        eps_av = mul(magic, er)
        mu_av = mul(magic, mr)
        n_av = sqrt(mul(mu_av, eps_av))
        # n = imag_check.apply(n)

        # eps_av = er
        # mu_av = mr
        # n = imag_check.apply(n)
        # n_av = n
        n_av = imag_check.apply(n_av)
        k = div(mul(w, n_av), c)
        z = sqrt(div(eps_av, mu_av))
        z = real_check.apply(z)

        M12_TE = 0.5 * 1j * mul((z - div(1,z)), (sin(mul(k, d))))
        M22_TE = cos(mul(k, d)) - 0.5 * 1j * mul((z + div(1,z)), (sin(mul(k, d))))
        # M12_TE = 0.5*1j*mul((mul(div(1, mu), div(k, k0)) + mul(mu, div(k0, k))), (sin(mul(k, d))))
        # M22_TE = cos(mul(k, d)) - 0.5*1j*mul((mul(div(1, mu), div(k, k0)) + mul(mu, div(k0, k))), (sin(mul(k, d))))
        r = div(M12_TE,M22_TE)
        t = div(1, M22_TE)
        # T = (mul(t, conj(t)).real).float()
        # R = (mul(r, conj(r)).real).float()

        R2 = sq(abs(div((mu_av - n_av), (mu_av + n_av))))
        r2 = div((mu_av - n_av), (mu_av + n_av))
        phiR = arctan(div(r2.imag,r2.real))
        phiR2 = arctan(div(2 * n_av.imag, (1 - sq(n_av.real) - sq(n_av.imag))))
        T2 = exp(-2 * mul(k.imag, d)) * sq((div(n_av, mu_av)).real * sq(abs(div(2 * mu_av, (n_av + mu_av)))))
        T3 = mul(exp(-2 * mul(k.imag, d)), sq(
            (div(mul(n_av.real, mu.real)
                 + mul(n_av.imag, mu.imag), (sq(mu.real) +
                                          sq(mu.imag)))) * m0 * sq(abs(div(2 * mu_av, (n_av + mu_av))))))
        t2 = exp(-1 * mul(k.imag, d)) * div(2*mr,(n+mr)) * sqrt(div(n , mr))
        phiT2 = arctan(div(t2.imag, t2.real))

        self.w = f
        self.k0 = k0
        self.k = k
        self.n = n
        self.n_av = n_av
        self.z = z
        self.eps = er
        self.eps_av = eps_av
        self.mu = mr
        self.mu_av = mu_av
        self.M12 = M12_TE
        self.M22 = M22_TE
        self.r = r
        self.t = t
        self.r2 = r2
        self.t2 = t2
        self.T = sq(abs(t))
        self.R = sq(abs(r))
        self.R2 = R2
        self.phiR2 = phiR2
        self.T2 = T2
        self.T3 = T3
        self.phiT2 = phiT2
        self.theta = theta
        self.magic = magic


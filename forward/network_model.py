"""
This is the module where the model is defined. It uses the nn.Module as a backbone to create the network structure
"""

import math
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt, square, cos, sin, conj, abs, tan
from utils.optical_constants import matrix_method_slab, lorentzian

class LorentzDNN(nn.Module):
    def __init__(self, flags):
        super(LorentzDNN, self).__init__()
        self.flags = flags


        # Create the constant for mapping the frequency w
        w_numpy = np.arange(flags.freq_low, flags.freq_high,
                            (flags.freq_high - flags.freq_low) / self.flags.num_spec_points)

        # Create eps_inf variable, currently set to a constant value
        # self.epsilon_inf = torch.tensor([5+0j],dtype=torch.cfloat)

        # Create the frequency tensor from numpy array, put variables on cuda if available
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.w = torch.tensor(w_numpy).cuda()
            # self.d = torch.tensor([0.5], requires_grad=True).cuda()
        else:
            self.w = torch.tensor(w_numpy)
            # self.d = torch.tensor([0.5], requires_grad=True)

        """
        General layer definitions:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1], bias=True))
            # torch.nn.init.uniform_(self.linears[ind].weight, a=1, b=2)

            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1], track_running_stats=True, affine=True))

        layer_size = flags.linear[-1]

        # Last layer is the Lorentzian parameter layer
        self.eps_w0 = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.eps_wp = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.eps_g = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.eps_inf = nn.Linear(layer_size, 1, bias=True)
        self.mu_w0 = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.mu_wp = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.mu_g = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.mu_inf = nn.Linear(layer_size, 1, bias=True)


    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G

        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            #print(out.size())
            if ind < len(self.linears) - 0:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
            else:
                out = bn(fc(out))

        e_w0 = F.relu(self.eps_w0(F.relu(out)))
        e_wp = F.relu(self.eps_wp(F.relu(out)))
        e_g = F.relu(self.eps_g(F.relu(out)))
        e_inf = F.relu(self.eps_inf(F.relu(out)))

        m_w0 = F.relu(self.mu_w0(F.relu(out)))
        m_wp = F.relu(self.mu_wp(F.relu(out)))
        m_g = F.relu(self.mu_g(F.relu(out)))
        m_inf = F.relu(self.mu_inf(F.relu(out)))

        # d = self.d(F.relu(out))

        self.eps_params_out = [e_w0, e_wp, e_g, e_inf]
        self.mu_params_out = [m_w0, m_wp, m_g, m_inf]

        # w0_out = w0
        # wp_out = wp
        # g_out = g
        # eps_inf_out = eps_inf
        # mu_inf_out = mu_inf

        # for p in (e_w0,e_wp,e_g,m_w0,m_wp,m_g):
        #     p = p.unsqueeze(2)

        # Expand them to parallelize, (batch_size, #osc, #spec_point)
        e_w0 = e_w0.unsqueeze(2).expand(out.size()[0], self.flags.num_lorentz_osc, self.flags.num_spec_points)
        e_wp = e_wp.unsqueeze(2).expand_as(e_w0)
        e_g = e_g.unsqueeze(2).expand_as(e_w0)
        m_w0 = m_w0.unsqueeze(2).expand_as(e_w0)
        m_wp = m_wp.unsqueeze(2).expand_as(e_w0)
        m_g = m_g.unsqueeze(2).expand_as(e_w0)

        # for p in (e_wp,e_g,m_w0,m_wp,m_g):
        #     p = p.expand_as(e_w0)

        w_expand = self.w.expand_as(e_w0)
        w_2 = self.w.expand(out.size()[0],self.flags.num_spec_points)

        # Define dielectric functions

        e1, e2 = lorentzian(w_expand, e_w0, e_wp, e_g)
        mu1, mu2 = lorentzian(w_expand, m_w0, m_wp, m_g)
        e1 = torch.sum(e1, 1).type(torch.cfloat)
        e2 = torch.sum(e2, 1).type(torch.cfloat)
        eps_inf = e_inf.expand_as(e1).type(torch.cfloat)
        e1 += eps_inf
        mu1 = torch.sum(mu1, 1).type(torch.cfloat)
        mu2 = torch.sum(mu2, 1).type(torch.cfloat)
        mu_inf = m_inf.expand_as(mu1).type(torch.cfloat)
        mu1 += mu_inf
        j = torch.tensor([0+1j],dtype=torch.cfloat).expand_as(e2)
        if torch.cuda.is_available():
            j = j.cuda()

        eps = add(e1, mul(e2,j))
        mu = add(mu1, mul(mu2, j))

        # n0 = sqrt(mul(mu,eps))
        n = sqrt(mul(mu, eps))
        # z = div(mu, n)
        n1 = n.real.type(torch.cfloat)
        n2 = n.imag.type(torch.cfloat)

        # TODO Initialize d to be cylinder height, but let it be a variable
        d_in = G[:, 1]
        if self.flags.normalize_input:
            d_in = d_in * 0.5 * (self.flags.geoboundary[5]-self.flags.geoboundary[1]) + (self.flags.geoboundary[5]+self.flags.geoboundary[1]) * 0.5

        self.d_out = d_in
        d = d_in.unsqueeze(1).expand_as(eps)
        # d = self.d.unsqueeze(1).expand_as(eps)

        # # Spatial dispersion
        # theta = 0.0033*mul(mul(w_2,d),n0).type(torch.cfloat)
        # magic = mul(tan(0.5*theta),0.5*theta).type(torch.cfloat)
        # eps = mul(magic,eps)
        # mu = mul(magic, mu)
        # n = sqrt(mul(mu, eps))

        self.eps_out = eps
        self.mu_out = mu
        self.n_out = n

        # self.test_var = n

        # r, t = matrix_method_slab(eps, mu, d, w_2)
        # return r, t

        alpha = torch.exp(-0.0033 * 4 * math.pi * mul(mul(d, abs(n.imag)), w_2))
        # print(alpha)
        T = mul(div(4 * n.real, add(square(n.real + 1), square(n.imag))), alpha).float()
        # R = square(div(abs(add(n.real,abs(n.imag))-1),abs(add(n.real,abs(n.imag))+1)))
        R = square(div(abs(n-1),abs(n+1)))
        return R,T



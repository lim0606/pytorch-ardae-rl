import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def logprob_gaussian(mu, logvar, z, do_unsqueeze=True, do_mean=True):
    '''
    Inputs:⋅
        z: b1 x nz
        mu, logvar: b2 x nz
    Outputs:
        prob: b1 x nz
    '''
    if do_unsqueeze:
        z = z.unsqueeze(1)
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)

    neglogprob = (z - mu)**2 / logvar.exp() + logvar + math.log(2.*math.pi)
    logprob = - neglogprob*0.5

    if do_mean:
        assert do_unsqueeze
        logprob = torch.mean(logprob, dim=1)

    return logprob


'''
copied and modified from https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
'''
def get_covmat(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
            rowvar == True:  m: dim x batch_size
            rowvar == False: m: batch_size x dim
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

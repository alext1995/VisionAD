import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

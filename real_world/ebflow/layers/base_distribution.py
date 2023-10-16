# Code from https://github.com/akandykeller/SelfNormalizingFlows
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class NegativeLogGaussian(nn.Module):
    """
    Standard Normal Likelihood (negative)
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.dim = dim = int(np.prod(size))
        self.N = MultivariateNormal(torch.zeros(dim, device='cuda'),
                                    torch.eye(dim, device='cuda'))

    def forward(self, input, context=None):
        return -self.log_prob(input, context).sum(-1)

    def log_prob(self, input, context=None, sum=True):
        try: 
            p = self.N.log_prob(input.view(-1, self.dim))
        except RuntimeError:
            p = self.N.log_prob(input.reshape(-1, self.dim))
        return p

    def sample(self, n_samples, context=None):
        x = self.N.sample((n_samples,)).view(n_samples, *self.size)
        log_px = self.log_prob(x, context)
        return x, log_px
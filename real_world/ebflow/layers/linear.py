# Code from https://github.com/akandykeller/SelfNormalizingFlows
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ebflow.layers.flowlayer import LinearFlowLayer
from ebflow.utils.toeplitz import get_sparse_toeplitz, get_toeplitz_idxs

import numpy as np

class Convolution(LinearFlowLayer):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 bias=True, stride=1, padding=0, dilation=1, groups=1, spec_norm=False, neg_init=False):

        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias
        self.spec_norm = spec_norm
        self.neg_init = neg_init

        self.model = torch.nn.Conv2d(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.use_bias)
        self.init_parameters()

    def init_parameters(self):
        self.logabsdet_dirty = True
        self.T_idxs, self.f_idxs = None, None

        w_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        
        w_init = torch.nn.init.dirac_(torch.empty(w_shape))
        w_noise = torch.nn.init.xavier_normal_(torch.empty(w_shape), gain=0.01)
        b_init = torch.nn.init.normal_(torch.empty(self.out_channels), std=w_noise.std())
        
        self.model.weight = torch.nn.Parameter(w_init*-1) if self.neg_init else torch.nn.Parameter(w_init)
        self.model.bias = torch.nn.Parameter(b_init) if self.use_bias else None
        if self.spec_norm:
            self.model = torch.nn.utils.spectral_norm(self.model)

    def forward(self, input, context=None, zero_ldj=False):
        if self.training:
            self.logabsdet_dirty = True
        self.output = self.model(input)
        ldj = 0. if zero_ldj else self.logdet(input, context)

        return self.output, ldj

    def reverse(self, input, context=None):
        if self.use_bias:
            bias = self.model.bias
            input = input - bias.view(1, -1, 1, 1)

        T_sparse = self.sparse_toeplitz(input, context)
        rev = torch.matmul(T_sparse.to_dense().inverse().to(input.device),
                        input.flatten(start_dim=1).unsqueeze(-1))
        rev = rev.view(input.shape)
        return rev

    def sparse_toeplitz(self, input, context=None):
        weight = self.model.weight
        if self.T_idxs is None or self.f_idxs is None:
            self.T_idxs, self.f_idxs = get_toeplitz_idxs(
                weight.shape, input.shape[1:], self.stride, self.padding)

        T_sparse = get_sparse_toeplitz(weight, input.shape[1:],
                                       self.T_idxs, self.f_idxs)
        return T_sparse

    def logdet(self, input, context=None):
        if self.logabsdet_dirty:
            T_sparse = self.sparse_toeplitz(input, context)
            self.logabsdet = torch.slogdet(T_sparse.to_dense())[1].to(input.device)
            self.logabsdet_dirty = False
        return self.logabsdet.view(1).expand(len(input))

class FullyConnected(Convolution):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, (1, 1), bias, **kwargs)

    def init_parameters(self):
        self.logabsdet_dirty = True
        self.T_idxs, self.f_idxs = None, None

        w_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        sq_c = min(self.out_channels, self.in_channels)

        w_init = torch.nn.init.zeros_(torch.empty(w_shape))
        w_eye = torch.nn.init.dirac_(torch.empty((sq_c, sq_c, *self.kernel_size)))
        w_init[:sq_c, :sq_c, :, :] += w_eye
        b_init = torch.nn.init.ones_(torch.empty(self.out_channels))

        self.model.weight = torch.nn.Parameter(w_init*-1) if self.neg_init else torch.nn.Parameter(w_init)
        self.model.bias = torch.nn.Parameter(b_init) if self.use_bias else None
        if self.spec_norm:
            self.model = torch.nn.utils.spectral_norm(self.model)

    def forward(self, input, context=None, zero_ldj=False):
        input = input.view(-1, self.in_channels, 1, 1)
        output, ldj = super().forward(input, context, zero_ldj=zero_ldj)
        return output.view(-1, self.out_channels), ldj

    def reverse(self, input, context=None):
        input = input.view(-1, self.out_channels, 1, 1)
        
        if self.use_bias:
            input = input - self.model.bias.view(1, -1, 1, 1)

        rev = torch.matmul(self.model.weight[:,:,0,0].inverse(),
                        input.flatten(start_dim=1).unsqueeze(-1))
        rev = rev.view(-1, self.in_channels)
        
        return rev

    def logdet(self, input, context=None):
        if self.in_channels != self.out_channels:
            return torch.tensor(0.0).expand(len(input)).to(input.device)
        if self.logabsdet_dirty:
            self.logabsdet = torch.slogdet(self.model.weight[:,:,0,0])[1]
            self.logabsdet_dirty = False
        return self.logabsdet.view(1).expand(len(input))
    
class Conv1x1(LinearFlowLayer):
    def __init__(self, n_channels, spec_norm=False):
        super().__init__()
        self.n_channels = n_channels
        
        w_np = np.random.randn(n_channels, n_channels)
        q_np = np.linalg.qr(w_np)[0]
        w_init = torch.from_numpy(q_np.astype('float32'))
        self.spec_norm = spec_norm

        self.model = torch.nn.Conv2d(in_channels=self.n_channels,
                                      out_channels=self.n_channels,
                                      kernel_size=(1,1),
                                      stride=1,
                                      padding=0,
                                      dilation=1,
                                      groups=1,
                                      bias=False)
        self.model.weight = torch.nn.Parameter(w_init.view(self.n_channels, self.n_channels, 1, 1))
        if self.spec_norm:
            self.model = torch.nn.utils.spectral_norm(self.model)

    def forward(self, x, context=None, zero_ldj=False):
        assert len(x.size()) == 4
        _, _, H, W = x.size()

        w = self.model.weight.view(self.n_channels, self.n_channels)
        ldj = 0 if zero_ldj else H * W * torch.slogdet(w)[1]
        z = self.model(x)

        return z, ldj

    def reverse(self, z, context=None):
        w = self.model.weight.view(self.n_channels, self.n_channels)
        w_inv = torch.inverse(w)
        w_inv = w_inv.view(self.n_channels, self.n_channels, 1, 1)

        x = F.conv2d(z, w_inv, bias=None, stride=1, padding=0,
                     dilation=1, groups=1)

        return x

    def logdet(self, input, context=None):
        raise NotImplementedError
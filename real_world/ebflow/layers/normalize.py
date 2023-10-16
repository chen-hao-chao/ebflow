# Code from https://github.com/akandykeller/SelfNormalizingFlows
import torch
from .flowlayer import LinearFlowLayer

class Normalization(LinearFlowLayer):
    def __init__(self, translation, scale, learnable=False):
        super().__init__()

        if learnable:
            self.translation = torch.nn.Parameter(torch.Tensor([translation]))
            self.scale = torch.nn.Parameter(torch.Tensor([scale]))
        else:
            self.register_buffer('translation', torch.Tensor([translation]))
            self.register_buffer('scale', torch.Tensor([scale]))

    def forward(self, input, context=None, zero_ldj=False):
        logdet = 0 if zero_ldj else self.logdet(input, context)
        return (input - self.translation) / self.scale, logdet
               
    def reverse(self, input, context=None):
        return (input * self.scale) + self.translation

    def logdet(self, input, context=None):
        if len(input.shape) == 4:
            N, C, H, W = input.size()
            logdet = -C * H * W * torch.log(self.scale)
        else:
            N, C = input.size()
            logdet = -C * torch.log(self.scale)
        return logdet.expand(N)

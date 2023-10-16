# Code from https://github.com/akandykeller/SelfNormalizingFlows
import torch
import torch.nn.functional as F
from .flowlayer import FlowLayer
import numpy as np

class LogitTransform(FlowLayer):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, context=None):
        return (1/self.alpha)*(torch.log(input) - torch.log(1-input)), self.logdet(input, context)

    def reverse(self, input, context=None):
        return torch.sigmoid(input*self.alpha)
    
    def logdet(self, input, context=None):
        return (-np.log(self.alpha) - torch.log(input) - torch.log(1-input)).flatten(start_dim=1).sum(-1)


class SigmoidTransform(FlowLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input, context=None):
        return torch.sigmoid(input), self.logdet(input, context)

    def reverse(self, input, context=None):
        return torch.log(input) - torch.log(1 - input)

    def logdet(self, input, context=None):
        log_derivative = F.logsigmoid(input) + F.logsigmoid(-input)
        return log_derivative.flatten(start_dim=1).sum(1)

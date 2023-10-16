# Code from https://github.com/akandykeller/SelfNormalizingFlows
import torch
from .flowlayer import FlowLayer

class Dequantization(FlowLayer):
    def __init__(self, deq_distribution):
        super(Dequantization, self).__init__()
        self.distribution = deq_distribution

    def forward(self, input, context=None):
        noise, log_qnoise = self.distribution.sample(input.size(0), input.float())
        return input + noise, -log_qnoise

    def reverse(self, input, context=None):
        return torch.clamp(input.floor(), 0, 255)

    def logdet(self, input, context=None):
        raise NotImplementedError
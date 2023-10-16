# Code from https://github.com/kamenbliznashki/normalizing_flows/blob/master/glow.py
from torch import nn

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u):
        for module in reversed(self):
            u = module.inverse(u)
        return u
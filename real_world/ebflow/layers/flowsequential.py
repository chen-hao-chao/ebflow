# Code from https://github.com/akandykeller/SelfNormalizingFlows
import torch
from .flowlayer import LinearFlowLayer

class FlowSequential(torch.nn.Module):
    def __init__(self, base_distribution, *modules):
        super().__init__()
        self.base_distribution = base_distribution
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
        self.sequence_modules = modules

    def __iter__(self):
        yield from self.sequence_modules
    
    def forward(self, input, context=None, zero_ldj=False):
        additional_logdet = 0
        for _, module in enumerate(self):
            if isinstance(module, LinearFlowLayer):
                input, layer_logdet = module(input, context=context, zero_ldj=zero_ldj)
            else:
                input, layer_logdet = module(input, context=context)
            additional_logdet += layer_logdet
        return input, self.base_distribution.log_prob(input)+additional_logdet

    def log_prob(self, input, context=None, zero_ldj=False):
        return self.forward(input, context=context, zero_ldj=zero_ldj)[1]

    def sample(self, n_samples, context=None):
        input, _ = self.base_distribution.sample(n_samples, context)
        for module in reversed(self.sequence_modules):
            input = module.reverse(input, context)
        return input

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    
    def reverse(self, input, context=None):        
        for module in reversed(self.sequence_modules):
                input = module.reverse(input, context)
        return input

class TransSequential(torch.nn.Module):
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
        self.sequence_modules = modules

    def __iter__(self):
        yield from self.sequence_modules

    def forward(self, input, context=None, zero_ldj=False):
        additional_logdet = 0
        for _, module in enumerate(self):
            if isinstance(module, LinearFlowLayer):
                input, layer_logdet = module(input, context=context, zero_ldj=zero_ldj)
            else:
                input, layer_logdet = module(input, context=context)
            additional_logdet += layer_logdet

        return input, additional_logdet

    def log_prob(self, input, context=None, zero_ldj=False):
        return self.forward(input, context, zero_ldj)[1]

    def reverse(self, input, context=None):        
        for module in reversed(self.sequence_modules):
            input = module.reverse(input, context)
        return input
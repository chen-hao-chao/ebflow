# Code from https://github.com/akandykeller/SelfNormalizingFlows
# Code from https://github.com/fissoreg/relative-gradient-jacobian
import torch
from .flowlayer import FlowLayer

class FlowActivationLayer(FlowLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input, context=None):
        act = self.activation(input, context)
        logdet = self.logdet(input, context)
        return act, logdet

    def act_prime(self, input, context=None):
        raise NotImplementedError()

    def logdet(self, input, context=None):
        logderiv = torch.log(torch.abs(self.act_prime(input, context)))
        return logderiv.flatten(start_dim=1).sum(dim=-1)

def newton_raphson_inverse(f, y, x0, context=None, n_iter=100):
    x = x0
    for _ in range(n_iter):
        fprime = torch.clamp(f.act_prime(x, context), min=1e-2)
        x = x - (f.activation(x, context) - y) / fprime
    return x

class SmoothLeakyRelu(FlowActivationLayer):
    def __init__(self, alpha=0.3, residual=False):
        super().__init__()
        self.alpha = alpha
        self.residual = residual

    def activation(self, input, context=None):
        alpha = self.alpha

        stacked = torch.stack((torch.zeros_like(input), input))
        lse = torch.logsumexp(stacked, dim=0)
        res = input if self.residual else 0
        return alpha * input + (1-alpha) * lse + res

    def act_prime(self, input, context=None):
        alpha = self.alpha
        res = 1 if self.residual else 0
        return alpha + (1-alpha) * torch.sigmoid(input) + res

    def reverse(self, input, context=None):
        y, x0 = input, input
        return newton_raphson_inverse(self, y, x0, context)

class SmoothTanh(FlowActivationLayer):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def activation(self, input, context=None):
        return torch.tanh(self.alpha * input) + self.beta * input

    def act_prime(self, input, context=None):
        return self.beta + self.alpha / torch.pow(torch.cosh(self.alpha * input), 2)

    def reverse(self, input, context=None):
        y, x0 = input, input
        return newton_raphson_inverse(self, y, x0, context)
    
class ActNorm(FlowActivationLayer):
    def __init__(self, n_dims):
        super().__init__()

        self.n_dims = n_dims
        self.translation = torch.nn.Parameter(torch.zeros(n_dims)).to('cuda')
        self.log_scale = torch.nn.Parameter(torch.zeros(n_dims)).to('cuda')
        self.register_buffer('initialized', torch.tensor(0))

    def forward(self, input, context=None, zero_ldj=False):
        reduce_dims = [i for i in range(len(input.size())) if i != 1]

        if not self.initialized:
            with torch.no_grad():
                mean = torch.mean(input, dim=reduce_dims)
                log_stddev = torch.log(torch.std(input, dim=reduce_dims) + 1e-8)
                self.translation.data.copy_(mean)
                self.log_scale.data.copy_(log_stddev)
                self.initialized.fill_(1)

        if len(input.size()) == 4:
            _, C, H, W = input.size()
            translation = self.translation.view(1, C, 1, 1)
            log_scale = self.log_scale.view(1, C, 1, 1)
        else:
            _, D = input.size()
            translation = self.translation.view(1, D)
            log_scale = self.log_scale.view(1, D)

        out = (input - translation) * torch.exp(-log_scale)
        ldj = 0. if zero_ldj else self.logdet(input, context)

        return out, ldj

    def reverse(self, input, context=None):
        assert self.initialized

        if len(input.size()) == 4:
            _, C, H, W = input.size()
            translation = self.translation.view(1, C, 1, 1)
            log_scale = self.log_scale.view(1, C, 1, 1)
        else:
            _, D = input.size()
            translation = self.translation.view(1, D)
            log_scale = self.log_scale.view(1, D)

        output = input * torch.exp(log_scale) + translation
        return output

    def act_prime(self, input, context=None):
        return torch.exp(-self.log_scale)

    def logdet(self, input, context=None):
        B = input.size(0)
        if len(input.size()) == 2:
            ldj = -self.log_scale.sum().expand(B)
        elif len(input.size()) == 4:
            H, W = input.size()[2:]
            ldj = -self.log_scale.sum().expand(B) * H * W

        return ldj

class LeakyRelu(FlowActivationLayer):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.fwd = torch.nn.LeakyReLU(negative_slope=alpha)
        self.rev = torch.nn.LeakyReLU(negative_slope=(1/alpha))

    def activation(self, input, context=None):
        return self.fwd(input)

    def act_prime(self, input, context=None):
        return torch.where(input > 0.0, 1.0, self.alpha)

    def reverse(self, input, context=None):
        return self.rev(input)
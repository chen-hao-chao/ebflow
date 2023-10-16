# Code modified from https://github.com/kamenbliznashki/normalizing_flows
import torch
from torch import nn
import copy
from torch.nn.parameter import Parameter
from torch.nn import init

class ActNorm(nn.Module):
    def __init__(self, input_size, return_logdet=True, dependency=False):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, input_size))
        self.scale = nn.Parameter(torch.ones(1, input_size))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.dependency = dependency

    def initialize(self, input):
        with torch.no_grad():
            mean = input.mean(dim=0)
            std = input.std(dim=0)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        if self.dependency:
            log_det = torch.log(torch.abs(self.scale)+1e-6).sum(dim=1)
        else:
            log_det = torch.zeros(input.shape[0]).to(input.device)
        return self.scale * (input + self.loc), log_det

    def inverse(self, output):
        return (output / self.scale) - self.loc

class Swish(nn.Module):
  def __init__(self, dim=-1):
    super().__init__()
    if dim > 0:
      self.beta = nn.Parameter(torch.ones((dim,)))
    else:
      self.beta = torch.ones((1,))

  def forward(self, x):
    if len(x.size()) == 2:
      return x * torch.sigmoid(self.beta[None, :] * x)
    else:
      return x * torch.sigmoid(self.beta[None, :, None, None] * x)

class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, weight_init='default', t_net_act='elu', norm_type="default"):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size, hidden_size)]
        for _ in range(n_hidden):
            s_net += [Swish(hidden_size), nn.Linear(hidden_size, hidden_size)]
        s_net += [Swish(hidden_size), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        if t_net_act != 'swish':
            if t_net_act == 'elu':
                for i in range(len(self.t_net)):
                    if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ELU()
            else:
                raise ValueError("t_net_act {} not recognized.".format(t_net_act))
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight.data)
        
        if weight_init == 'ortho':
            init_weights(self.s_net)
            init_weights(self.t_net)

    def forward(self, x):
        # apply mask
        mx = x * self.mask
        # run through model
        s = self.s_net(mx)
        t = self.t_net(mx)
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = (- (1 - self.mask) * s).sum(dim=1)  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u):
        # apply mask
        mu = u * self.mask
        # run through model
        s = self.s_net(mu)
        t = self.t_net(mu)
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        return x
        
class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-8):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        # log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        log_abs_det_jacobian = torch.zeros(y.shape[0], device=y.device)
        
        return y, log_abs_det_jacobian

    def inverse(self, y):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        return x

class Linear_Layer(nn.Module):
    """
    Linear layer and its inverse
    Source: 
    https://stackoverflow.com/questions/59878319/can-you-reverse-a-pytorch-neural-network-and-activate-the-inputs-from-the-output
    """
    def __init__(self, input_size, dependency=False, weight_init='default', norm_type="default"):
        super().__init__()
        self.dependency = dependency
        self.linear_layer = nn.Linear(input_size, input_size)
        
        def init_weights(m):
            torch.nn.init.orthogonal_(m.weight.data)
        
        if weight_init == 'ortho':
            init_weights(self.linear_layer)
        if norm_type == "weight_norm":
            self.linear_layer = torch.nn.utils.weight_norm(self.linear_layer)
        elif norm_type == "spectral_norm":
            self.linear_layer = torch.nn.utils.spectral_norm(self.linear_layer)

    def forward(self, x):
        y = self.linear_layer(x)
        if self.dependency:
            det = self.linear_layer.weight[0,0]*self.linear_layer.weight[1,1] - self.linear_layer.weight[0,1]*self.linear_layer.weight[1,0]
            log_det = torch.ones(y.shape[0]).to(y.device)*torch.log(torch.abs(det)+1e-8)
        else:
            log_det = torch.zeros(y.shape[0]).to(y.device)
        return y, log_det
    
    def inverse(self, y):
        y = y - self.linear_layer.bias[None, ...]
        y = y[..., None]  # 'torch.solve' requires N column vectors (i.e. shape (N, n, 1)).
        y = torch.solve(y, self.linear_layer.weight)[0]
        x = torch.squeeze(y)  # remove the extra dimension that we've added for 'torch.solve'.
        return x

class Masked_Linear_Layer(nn.Module):
    """
    Masked Linear layer and its inverse
    Source: 
    https://stackoverflow.com/questions/59878319/can-you-reverse-a-pytorch-neural-network-and-activate-the-inputs-from-the-output
    """
    def __init__(self, input_size, mask, dependency=False, one_init=False):
        super().__init__()
        self.dependency = dependency
        if one_init:
            w_init = torch.nn.init.ones_(torch.empty((input_size, input_size)))
        else:
            w_init = torch.nn.init.orthogonal_(torch.empty((input_size, input_size)))
        self.weights = torch.nn.Parameter(w_init)
        self.mask = mask

    def forward(self, x):
        matrix = self.weights*self.mask.to(x.device)
        y = torch.mm(x, matrix)
        if self.dependency:
            det = torch.linalg.det(matrix)
            log_det = torch.ones(y.shape[0]).to(y.device)*torch.log(torch.abs(det))
        else:
            log_det = torch.zeros(y.shape[0]).to(y.device)
        return y, log_det
    
    def inverse(self, y):
        matrix = self.weights*self.mask.to(y.device)
        y = y[..., None]  # 'torch.solve' requires N column vectors (i.e. shape (N, n, 1)).
        y = torch.solve(y, matrix)[0]
        x = torch.squeeze(y)  # remove the extra dimension that we've added for 'torch.solve'.
        return x

class ELU_Layer(nn.Module):
    def __init__(self, input_size, alpha=0.3):
        super().__init__()
        self.input_size = input_size
        self.alpha = alpha

    def act_prime(self, input):
        alpha = self.alpha
        return alpha + (1-alpha) * torch.sigmoid(input)
    
    def forward(self, x):
        y = self.alpha*x + (1 - self.alpha)*(torch.logaddexp(x, torch.zeros(x.shape, device=x.device)))
        log_det = torch.log( torch.abs( self.alpha + (1-self.alpha) * torch.sigmoid(x) ) + 1e-8 ).sum(dim=1)
        return y, log_det
    
    def inverse(self, input):
        with torch.no_grad():
            ''' Source: https://github.com/akandykeller/SelfNormalizingFlows/blob/9feebb36255c5947aec1c58acd91be79135aabe8/snf/layers/activations.py#L26'''
            y, x = input, input
            n_iter = 100
            for _ in range(n_iter):
                fprime = torch.clamp(self.act_prime(x), min=1e-2)
                f, _ = self.forward(x)
                x = x - (f - y) / fprime
            return x

class Tanh_Layer(nn.Module):
    def __init__(self, input_size, alpha=0.3):
        super().__init__()
        self.input_size = input_size
        self.alpha = alpha

    def act_prime(self, input):
        output = self.alpha / torch.pow(torch.cosh(self.alpha * input), 2)
        return output
    
    def forward(self, x):
        y = torch.tanh(self.alpha * x)
        log_det = torch.ones(y.shape[0]).to(y.device)*self.logdet(x)
        return y, log_det
    
    def inverse(self, input):
        with torch.no_grad():
            ''' Source: https://github.com/akandykeller/SelfNormalizingFlows/blob/9feebb36255c5947aec1c58acd91be79135aabe8/snf/layers/activations.py#L26'''
            y, x = input, input
            n_iter = 100
            for _ in range(n_iter):
                fprime = torch.clamp(self.act_prime(x), min=1e-2)
                f, _ = self.forward(x)
                x = x - (f - y) / fprime
            return x
        
    def logdet(self, input):
        logderiv = torch.log(torch.abs(self.act_prime(input)))
        return logderiv.flatten(start_dim=1).sum(dim=-1)


class Weight_Multiply_Layer(nn.Module):
    def __init__(self, input_size, dependency=False):
        super().__init__()
        self.dependency = dependency
        self.weight = Parameter(torch.empty((input_size, input_size)))
        init.kaiming_normal_(self.weight, mode='fan_in')

    def forward(self, x):
        y = torch.mm(x, self.weight.t())
        if self.dependency:
            det = torch.linalg.det(self.weight)
            log_det = 0.5*torch.log(det**2+1e-8)
        else:
            log_det = torch.zeros(y.shape[0]).to(y.device)
        return y, log_det
    
    def inverse(self, y):
        y = y[..., None]  # 'torch.solve' requires N column vectors (i.e. shape (N, n, 1)).
        y = torch.solve(y, self.weight.t())[0]
        x = torch.squeeze(y)  # remove the extra dimension that we've added for 'torch.solve'.
        return x

import torch
from torch import nn
import numpy as np
from ebflow.models.layers import Linear_Layer, LinearMaskedCoupling, ActNorm
from ebflow.models.utils import FlowSequential

class Net(nn.Module):
  def __init__(self, input_size=2, n_blocks=10, hid=64, eps=1e-8, sigma=1, dependency=False, weight_init='default', t_net_act='elu', norm_type="default"):
    super().__init__()
    self.sigma = sigma
    self.eps = eps
    self.input_size = input_size
    self.dependency = dependency
    mask = torch.arange(input_size).float() % 2
    modules = nn.ModuleList()
    for i in range(n_blocks):
        modules.append(Linear_Layer(input_size=input_size, dependency=dependency, weight_init=weight_init, norm_type=norm_type))
        modules.append(LinearMaskedCoupling(input_size=input_size, hidden_size=hid, n_hidden=2, mask=mask, weight_init=weight_init, t_net_act=t_net_act, norm_type=norm_type))
        modules.append(ActNorm(input_size=input_size, dependency=dependency))
        mask = 1 - mask
    modules.append(Linear_Layer(input_size=input_size, dependency=dependency, weight_init=weight_init, norm_type=norm_type))
    self.layers = FlowSequential(*modules)
  
  def forward_ith_layer(self, u, i, last=False):
    layer_out = u
    log_det_all = 0
    blocks = 1 if last else 3
    for j in range(blocks): # linear-coupling-actnorm
      layer_out, log_det = self.layers[i*3+j](layer_out)
      log_det_all = log_det_all + log_det
    
    return layer_out, log_det_all
  
  def get_weights_ith_layer(self, i, last=False):
    blocks = 1 if last else 3
    weights = None
    for j in range(blocks): # linear-coupling-actnorm
      layer = self.layers[i*3+j]
      if isinstance(layer, Linear_Layer):
          weights = torch.cat((weights, layer.linear_layer.weight.flatten()), 0) if weights is not None else layer.linear_layer.weight.flatten()
      if isinstance(layer, LinearMaskedCoupling):
          for layer_number in range(len(layer.t_net)):
              layer_ = layer.t_net[layer_number]
              if isinstance(layer_, nn.Linear):
                  weights = torch.cat((weights, layer_.weight.flatten()), 0) if weights is not None else layer_.weight.flatten()
          for layer_number in range(len(layer.s_net)):
              layer_ = layer.s_net[layer_number]
              if isinstance(layer_, nn.Linear):
                  weights = torch.cat((weights, layer_.weight.flatten()), 0) if weights is not None else layer_.weight.flatten()
    return weights.flatten().detach().cpu().numpy()

  def inverse(self, u):
    return self.layers.inverse(u)

  def neg_energy(self, x):
    x = x.requires_grad_()
    layer_out, log_abs_det = self.layers(x)
    log_q = -np.log( 2 * (np.pi**(self.input_size*0.5)) * (self.sigma**(self.input_size)) ) - torch.sum(layer_out**2, dim=1) / (2*self.sigma**2)
    neg_e = log_q + log_abs_det
    return neg_e, log_q, log_abs_det
  
  def log_p(self, x):
    layer_out, log_abs_det = self.layers(x)
    log_q = -np.log( (2*self.sigma)**(self.input_size) ) - torch.sum(torch.abs(layer_out), dim=1)
    log_p = log_q + log_abs_det
    return log_p
  
  def inference(self, x):
    layer_out, _ = self.layers(x)
    return layer_out

  def score(self, x):
    x = x.requires_grad_()
    layer_out, log_abs_det = self.layers(x)
    log_q = -torch.sum(layer_out**2, dim=1) / (2*self.sigma**2)
    neg_e = log_q + log_abs_det
    score = torch.autograd.grad(neg_e.sum(), x, create_graph=True)[0]
    return score
  
  def open_dependency(self):
    for m in list(self.layers.children())[::-1]:
      if isinstance(m, Linear_Layer):
        m.dependency = True
      if isinstance(m, ActNorm):
        m.dependency = True
  
  def restore_dependency(self):
    for m in list(self.layers.children())[::-1]:
      if isinstance(m, Linear_Layer):
        m.dependency = self.dependency
      if isinstance(m, ActNorm):
        m.dependency = self.dependency
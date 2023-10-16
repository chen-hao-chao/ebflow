import torch
from torch import nn
import numpy as np
from ebflow.models.layers import Masked_Linear_Layer, ELU_Layer, ActNorm
from ebflow.models.utils import FlowSequential

class Net(nn.Module):
  def __init__(self, type="full", input_size=2, n_blocks=1, eps=1e-8, sigma=1, dependency=False):
    super().__init__()
    self.sigma = sigma
    self.eps = eps
    self.input_size = input_size
    self.dependency = dependency
    if type in ["full", "triu", "tril"]:
      if type == "full":
        mask = torch.ones((input_size,input_size)).to(dtype=torch.float32)
      elif type == "triu":
        mask = torch.triu(torch.ones((input_size,input_size))).to(dtype=torch.float32)
      elif type == "tril":
        mask = torch.tril(torch.ones((input_size,input_size))).to(dtype=torch.float32)
      modules = nn.ModuleList()
      for i in range(n_blocks):
          modules.append(Masked_Linear_Layer(input_size=input_size, mask=mask, dependency=dependency))
          modules.append(ELU_Layer(input_size=input_size))
          modules.append(ActNorm(input_size=input_size, dependency=dependency))
      modules.append(Masked_Linear_Layer(input_size=input_size, mask=mask, dependency=dependency))
    else:
      mask_l = torch.tril(torch.ones((input_size,input_size))).to(dtype=torch.float32)
      mask_u = torch.triu(torch.ones((input_size,input_size))).to(dtype=torch.float32)
      modules = nn.ModuleList()
      for i in range(n_blocks):
          modules.append(Masked_Linear_Layer(input_size=input_size, mask=mask_u, dependency=dependency))
          modules.append(Masked_Linear_Layer(input_size=input_size, mask=mask_l, dependency=dependency, one_init=True))
          modules.append(ELU_Layer(input_size=input_size))
          modules.append(ActNorm(input_size=input_size, dependency=dependency))
      modules.append(Masked_Linear_Layer(input_size=input_size, mask=mask_u, dependency=dependency))
      modules.append(Masked_Linear_Layer(input_size=input_size, mask=mask_l, dependency=dependency, one_init=True))
    self.layers = FlowSequential(*modules)

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
      if isinstance(m, Masked_Linear_Layer):
        m.dependency = True
      if isinstance(m, ActNorm):
        m.dependency = True
  
  def restore_dependency(self):
    for m in list(self.layers.children())[::-1]:
      if isinstance(m, Masked_Linear_Layer):
        m.dependency = self.dependency
      if isinstance(m, ActNorm):
        m.dependency = self.dependency
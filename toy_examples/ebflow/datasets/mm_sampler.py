import numpy as np
import torch
from absl import app
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

def fn(x, dist=2, k=5):
    '''
    x (torch tensor): [bs, dim]
    '''
    dtype = x.dtype
    y = torch.zeros(x.shape).to(x.device).to(dtype=dtype)
    y[:,0] = x[:,0].clone()
    for i in range(1, x.shape[1]):
        y[:,i] = torch.tanh( x[:,i]*k ) * ( y[:,i-1] + (dist*(2**i)) )
    return y

def inv_fn(y, dist=2, k=5, bound=1-1e-16):
    '''
    x (torch tensor): [bs, dim]
    '''
    dtype = y.dtype
    x = torch.zeros(y.shape).to(y.device).to(dtype=dtype)
    x[:,0] = y[:,0].clone()
    for i in range(1, y.shape[1]):
        x[:,i] = torch.atanh( torch.clamp(y[:,i] / ( y[:,i-1] + (dist*(2**i)) ), -bound, bound )) / k
    return x

def multimodal_prob(y):
    '''
    y (torch tensor): [bs, dim]
    '''
    dim = y.shape[1]
    x = inv_fn(y)
    Jacob_det = 1
    for i in range(dim):
        x_grad, = torch.autograd.grad(x[:, i].sum(), y, create_graph=True)
        Jacob_det *= x_grad[:, i]
    Jacob_det = torch.abs(1 / (4*3*(1 - (y[:, 1]/4)**2)+1e-16) )
    print(Jacob_det)
    p = torch.exp( -0.5*torch.sum(x**2, dim=1) ) / ( (2*np.pi)**(dim/2) )
    p = p * Jacob_det
    return p

def multimodal_score(y):
    '''
    y (torch tensor): [bs, dim]
    '''
    p = multimodal_prob(y)
    log_p = torch.log(p+1e-8)
    log_p_grad, = torch.autograd.grad(log_p.sum(), y)
    return log_p_grad
    

def plot():
    dim = 10
    num_samples = 50000
    w = 10
    device = torch.device('cuda:0')

    # Sampled points
    x = torch.randn((num_samples, dim), device=device).to(dtype=torch.double)
    u = fn(x).detach().clone().cpu().numpy()
    range_ = [[-w, w],[-w, w]]
    fig = figure(figsize=(6, 6), dpi=300)
    plt.hist2d(u[:,0], u[:,1], bins=100, range=range_, cmap=plt.cm.jet)
    plt.savefig("./fig/test2.png")
    plt.close(fig)

    # Sampled points
    for i in range(0, x.shape[1]):
        fig = figure(figsize=(6, 3), dpi=500)
        plt.hist(u[:,i], bins=1000)
        plt.savefig("./fig/"+str(i)+".png")
        plt.savefig("./fig/"+str(i)+".eps")
        plt.close(fig)

def main(argv):
    plot()

if __name__ == "__main__":
    app.run(main)
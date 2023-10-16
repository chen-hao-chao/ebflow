from re import X
import numpy as np
import torch
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)

def evaluate(x, model, score_oracle, prob_oracle):
    '''
    Inputs:
    x: np array
    model: ebflow model
    score_oracle: true score value for x
    ------
    Outputs:
    kld, fd
    '''
    model.open_dependency()
    model.eval().double()
    x = torch.tensor(torch.from_numpy(x), requires_grad=True).to('cuda')
    log_p = model.log_p(x)
    const = prob_oracle * np.log(prob_oracle)
    kld = (-log_p).detach().cpu().numpy() + const
    score = torch.autograd.grad(torch.sum(log_p), x)[0]
    score_estimate_np = score.detach().cpu().numpy()
    fd = np.sum((score_oracle - score_estimate_np)**2, axis=1)*0.5
    model.restore_dependency()
    return np.mean(kld), np.mean(fd)

def oracle_score_denominator(point, batch, sigma):
    return torch.exp(  - ( ( (batch[:, 0]-point[0])**2 + (batch[:, 1]-point[1])**2 )  / (2*(sigma**2)) )   )  /  ( 2*np.pi*(sigma**2) )

def guassian_prob(x, tx, sigma):
    return torch.exp(  - ( ( (tx[:, 0]-x[0])**2 + (tx[:, 1]-x[1])**2 )  / (2*(sigma**2)) )   )  /  ( 2*np.pi*(sigma**2) )  

def oracle_score_numerator(point, batch, sigma):
    p = guassian_prob(point, batch, sigma)
    diff = ( batch-point ) /  (sigma**2)
    diff[:, 0] *= p
    diff[:, 1] *= p
    return diff, p

def get_oracle_score(points, batch, sigma, eps=1e-8, output_prob=False):
    points = torch.tensor(points).to(batch.device)
    numerator, prob = oracle_score_numerator(points, batch, sigma = sigma)
    denominator = oracle_score_denominator(points, batch, sigma = sigma)
    sum_numerator = torch.sum(numerator, dim=0)
    sum_denominator = torch.sum(denominator, dim=0) 
    prob = torch.mean(prob, dim=0)
    if output_prob:
        return sum_numerator / (sum_denominator + eps), prob
    else:
        return sum_numerator / (sum_denominator + eps)

def plot_energy(config, energy, filename):
    w = config['width']
    h = config['height']

    grid_size = 100
    xx, yy = torch.meshgrid(torch.linspace(-w, w, grid_size), torch.linspace(-h, h, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zeros = torch.zeros((zz.shape[0], 2), device='cuda')
    zeros[:, 0:2] = zz
    zz = zeros

    energy.eval().float()
    energy.open_dependency()
    neg_energy, log_q, log_abs_det = energy.neg_energy(zz)
    energy.restore_dependency()

    neg_energy = torch.exp(neg_energy.to('cpu').view(*xx.shape))
    neg_energy[torch.isnan(neg_energy)] = 0
    log_q = torch.exp(log_q.to('cpu').view(*xx.shape))
    log_q[torch.isnan(log_q)] = 0
    log_abs_det = torch.exp(log_abs_det.to('cpu').view(*xx.shape))
    log_abs_det[torch.isnan(log_abs_det)] = 0

    
    plt.pcolormesh(xx, yy, neg_energy.data.numpy())
    plt.gca().set_aspect('equal', 'box')
    plt.axis('off')
    plt.savefig(filename)
    plt.clf()
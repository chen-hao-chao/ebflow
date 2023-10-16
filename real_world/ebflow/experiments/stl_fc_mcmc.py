import torch
import json
import os
import torchvision
import random
import numpy as np

from ebflow.layers.ema import ExponentialMovingAverage
from ebflow.models.stl.fc import create_model

def set_deterministic(seed):
    # OS
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # Pytorch
    torch.manual_seed(seed)
    # Random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)

def main(args):
    # Read the config file
    config_file = os.path.join('ebflow', 'configs', \
                               args.config.split('_')[0], args.config.split('_')[1], args.loss+'.txt')
    data = open(config_file).read()
    config = json.loads(data)

    # Deterministic generation
    set_deterministic(seed=config['seed'])

    # Disable MaP
    config['MaP'] = 0 if args.withoutMaP else 1

    # Load the model's weights from 'restore_path'
    config['restore_path'] = args.restore_path

    # Model with preprocessing function
    model, trans = create_model(num_layers=config['num_layers'],
                                logit_smoothness=config['logit_smoothness'],
                                alpha=config['alpha'],
                                bias=bool(config['bias']),
                                neg_init=bool(config['neg_init']),
                                MaP=bool(config['MaP']))
    model.to('cuda')
    trans.to('cuda')
      
    # Exponential Moving Average (EMA)
    if config['ema_decay'] >= 0:
        ema = ExponentialMovingAverage(model.parameters(), decay=config['ema_decay'])
    else:
        ema = None

    # Load checkpoint
    if config['restore_path']:
        checkpoint = torch.load(config['restore_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema'])
    else:
        print("ERROR")
        assert False
    
    ema.copy_to(model.parameters())

    # Update paths and create directories
    try:
        os.makedirs("results_inpainting", exist_ok = True)
        print("Directory created successfully")
    except OSError as error:
        print("Directory can not be created")

    # Perform the inpainting task
    inpainting(config, model, trans)

def inpainting(config, model, trans):
    b = torchvision.transforms.GaussianBlur(int(config['sampling_kernel_size']))
    from ebflow.datasets.stl import load_data
    test_loader, _, _ = load_data(batch_size=config['sampling_batch_size'])

    from ebflow.datasets.mnist import load_data
    test_loader_crop, _, _ = load_data(batch_size=config['sampling_batch_size'], size=64)

    for x_crop_, _ in test_loader_crop:
        x_crop_in = torch.where(x_crop_ > 0.5, 1.0, 0.0).float().to('cuda')
        x_crop_out = torch.where(x_crop_ > 0.5, 0.0, 1.0).float().to('cuda')
        break

    for x_, _ in test_loader:
        with torch.no_grad():
            # create mask
            x_ = x_.float().to('cuda')
            mask = torch.randn(x_.shape).to(x_.device)
            mask = trans.reverse(mask)
            x_ = x_*x_crop_out + mask*x_crop_in
            x, _ = trans.forward(x_)
        
        snr = config['sampling_snr']
        step_size = config['sampling_step_size']
        smooth_fac = config['sampling_smooth_fac']
        num_iter = int(config['sampling_iteration'])
        for i in range(num_iter):
            with torch.no_grad():
                x_b_diff = (b(x) - x) * x_crop_in
            x.requires_grad_()
            log_p = model.log_prob(x, zero_ldj=True)  
            score = torch.autograd.grad(log_p.sum(), x, create_graph=False)[0]
            noise = torch.randn(score.shape).to('cuda')
            score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            ratio = (noise_norm / score_norm)**2
            move = score*x_crop_in*(step_size**2/2)
            x_mean = x + move + smooth_fac*x_b_diff
            x_noise = x_mean + (snr*ratio)*noise*x_crop_in*step_size
            x = x_mean.detach() if i == (num_iter-1) else x_noise.detach()

        img = trans.reverse(x)
        torchvision.utils.save_image(img / 256., "results_inpainting/inpainting_fc_stl.png", nrow=10, padding=2, normalize=False)
        break
        
    print("finish")
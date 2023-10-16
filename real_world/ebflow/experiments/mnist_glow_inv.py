import torch
import json
import os
import torchvision
import random
import numpy as np

from ebflow.layers.ema import ExponentialMovingAverage
from ebflow.models.mnist.glow import create_model


# set_deterministic
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
    # Deterministic generation
    set_deterministic(seed=0)

    # Read the config file
    config_file = os.path.join('ebflow', 'configs', \
                               args.config.split('_')[0], args.config.split('_')[1], args.loss+'.txt')
    data = open(config_file).read()
    config = json.loads(data)

    # Disable MaP
    config['MaP'] = 0 if args.withoutMaP else 1

    # Load the model's weights from 'restore_path'
    config['restore_path'] = args.restore_path
    
    # Model with preprocessing function
    model, trans = create_model(num_blocks=config['num_blocks'],
                                block_size=config['block_size'], 
                                logit_smoothness=config['logit_smoothness'],
                                MaP=bool(config['MaP']))
    model.to('cuda')
    trans.to('cuda')

    # ($) Exponential Moving Average (EMA)
    if config['ema_decay'] >= 0:
        ema = ExponentialMovingAverage(model.parameters(), decay=config['ema_decay'])
    else:
        ema = None
    
    if config['restore_path'] is not '':
        checkpoint = torch.load(config['restore_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema'])
    else:
        print("ERROR")
        assert False
    
    ema.copy_to(model.parameters())

    with torch.no_grad():
        x_sample = model.sample(100)
        x_sample = trans.reverse(x_sample)
        torchvision.utils.save_image(x_sample / 256., "inverse_generation.png", nrow=10, padding=2, normalize=False)
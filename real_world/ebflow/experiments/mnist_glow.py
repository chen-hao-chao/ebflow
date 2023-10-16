import torch
import json
import os

from ebflow.train.experiment import Experiment
from ebflow.layers.ema import ExponentialMovingAverage
from ebflow.datasets.mnist import load_data
from ebflow.models.mnist.glow import create_model

def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for config, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        elif config.endswith(".W"):
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def main(args):
    # Read the config file
    config_file = os.path.join('ebflow', 'configs', \
                               args.config.split('_')[0], args.config.split('_')[1], args.loss+'.txt')
    data = open(config_file).read()
    config = json.loads(data)
    
    # Data loaders
    train_loader, val_loader, test_loader = load_data(data_aug=bool(config['data_aug']), batch_size=config['batch_size'])

    # Model with preprocessing function
    model, trans = create_model(num_blocks=config['num_blocks'],
                                block_size=config['block_size'], 
                                logit_smoothness=config['logit_smoothness'],
                                MaP=bool(config['MaP']))
    model.to('cuda')
    trans.to('cuda')
                        
    # Exponential Moving Average (EMA)
    if config['ema_decay'] >= 0:
        ema = ExponentialMovingAverage(model.parameters(), decay=config['ema_decay'])
    else:
        ema = None

    # Optimizer
    optimizer = torch.optim.Adam(add_weight_decay(model, weight_decay=config['wd']),
                                                        lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(1,config['milestones'])], gamma=config['gamma'])
    
    # Runner
    experiment = Experiment(model, trans, ema, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)
    experiment.run()
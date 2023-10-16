import torch
import json
import os

from ebflow.train.experiment import Experiment
from ebflow.layers.ema import ExponentialMovingAverage
from ebflow.datasets.cifar10 import load_data
from ebflow.models.cifar.cnn import create_model

def main(args):
    # Read the config file
    config_file = os.path.join('ebflow', 'configs', \
                               args.config.split('_')[0], args.config.split('_')[1], args.loss+'.txt')
    data = open(config_file).read()
    config = json.loads(data)

    # Disable MaP
    config['MaP'] = 0 if args.withoutMaP else 1

    # Data loaders
    train_loader, val_loader, test_loader = load_data(data_aug=bool(config['data_aug']), batch_size=config['batch_size'])

    # Model with preprocessing function
    model, trans = create_model(num_layers=config['num_layers'], 
                                num_blocks=config['num_blocks'],
                                kernel_size=config['kernel_size'], 
                                logit_smoothness=config['logit_smoothness'],
                                alpha=config['alpha'],
                                bias=bool(config['bias']),
                                neg_init=bool(config['neg_init']),
                                spec_norm=bool(config['spec_norm']),
                                MaP=bool(config['MaP']))
    model.to('cuda')
    trans.to('cuda')
                        
    # Exponential Moving Average (EMA)
    if config['ema_decay'] >= 0:
        ema = ExponentialMovingAverage(model.parameters(), decay=config['ema_decay'])
    else:
        ema = None

    # Optimizer
    optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=config['lr'],
                                        weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(1,config['milestones'])], gamma=config['gamma'])
    
    # Runner
    experiment = Experiment(model, trans, ema, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)
    experiment.run()
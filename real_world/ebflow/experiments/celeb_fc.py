import torch
import json
import os

from ebflow.train.experiment import Experiment
from ebflow.layers.ema import ExponentialMovingAverage
from ebflow.datasets.celebA import load_data
from ebflow.models.celeb.fc import create_model

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

    # Optimizer
    optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=config['lr'],
                                        weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(1,config['milestones'])], gamma=config['gamma'])

    # Path
    config['restore_path'] = args.restore_path

    # Runner
    experiment = Experiment(model, trans, ema, train_loader, val_loader, test_loader,
                            optimizer, scheduler, datasize=(3,64,64), **config)
    experiment.run()
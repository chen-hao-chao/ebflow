import torch
import json
import os

from ebflow.train.experiment import Experiment
from ebflow.layers.ema import ExponentialMovingAverage
from ebflow.datasets.mnist import load_data
from ebflow.models.mnist.fc import create_model

def main(args):
    # Read the config file
    config_file = os.path.join('ebflow', 'configs', \
                               args.config.split('_')[0], args.config.split('_')[1], args.loss+'.txt')
    data = open(config_file).read()
    config = json.loads(data)

    # Disable MaP
    config['MaP'] = 0 if args.withoutMaP else 1
    # Disable eval_only mode
    config['eval_only'] = 1 if args.eval_only else 0

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
    
    if config['eval_only'] == 1:
        # Load the model's weights from 'restore_path'
        config['restore_path'] = args.restore_path
        
        if config['restore_path'] is not '':
            checkpoint = torch.load(config['restore_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            ema.load_state_dict(checkpoint['ema'])
        else:
            print("ERROR")
            assert False
        
        ema.copy_to(model.parameters())

    # Optimizer
    optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=config['lr'],
                                        weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(1,config['milestones'])], gamma=config['gamma'])
    
    # Runner
    experiment = Experiment(model, trans, ema, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)
    experiment.run()
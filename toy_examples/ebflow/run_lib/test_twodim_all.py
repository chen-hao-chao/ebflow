import torch
import numpy as np
from absl import app
import os
import json
import scipy.stats

from ebflow.datasets.datasets import get_dataset
from ebflow.models import coupling_net
from ebflow.models.ema import ExponentialMovingAverage
from ebflow.run_lib.utils import get_oracle_score, evaluate, plot_energy

def mean_confidence_interval(data, confidence=0.95):
    '''
    Source: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    '''
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def run_test(config):
    dataset = config['dataset']
    # Build data iterators
    config['batch_size'] = 10000
    test_ds, _ = get_dataset(config)
    test_iter = iter(test_ds)
    # Get an eval batch
    data = next(test_iter)
    batch_eval = torch.from_numpy(data['position']._numpy()).to('cuda', dtype=torch.double)
    std = torch.empty(batch_eval.shape, device=batch_eval.device).fill_(config['std'])
    noise = torch.randn_like(batch_eval, device=batch_eval.device)
    batch_eval_x = (batch_eval + std * noise).detach().cpu().numpy()
    
    # (Evaluation) Calculate Oracle Score
    score_oracle = np.zeros(batch_eval_x.shape)
    prob_oracle = np.zeros(batch_eval_x.shape[0])
    for i in range(batch_eval_x.shape[0]):
        score_pt, prob_pt = get_oracle_score(batch_eval_x[i], batch_eval, config['std'], output_prob=True)
        score_oracle[i, :] = score_pt.cpu().numpy()
        prob_oracle[i] = prob_pt.cpu().numpy()

    for loss in ['ml', 'sml', 'ssm', 'dsm', 'fdssm']:
        # Read the data from the file
        config_file = os.path.join('ebflow', 'configs', dataset, loss+'.txt')
        data = open(config_file).read()
        config = json.loads(data)
        print("Loss: {}".format(loss))

        # Eval
        klds = []
        fds = []
        for i in range(3):
            # Model Setups
            ebflow = coupling_net.Net(n_blocks=config['n_blocks'],
                                    hid=config['hidden_size'],
                                    input_size=config['input_size'],
                                    weight_init=config['weight_init'],
                                    t_net_act=config['t_net_act'],
                                    dependency=bool(config['dependency']),
                                    norm_type=config['norm_type'],
                                    sigma=config['sigma']).to('cuda', dtype=torch.double)
            
            # Exponential Moving Average (EMA)
            ema = ExponentialMovingAverage(ebflow.parameters(), decay=0.999)
                
            ckpt = '100000' if dataset == "checkerboard" else '50000'
            checkpoint_file = "results_toy_new/"+dataset+"/"+loss+"/"+str(i+1)+"/checkpoints/checkpoint_"+ckpt+".pth"
            checkpoint = torch.load(checkpoint_file, map_location='cuda')
            ebflow.load_state_dict(checkpoint['model'])
            ema.load_state_dict(checkpoint['ema'])
            ema.copy_to(ebflow.parameters())
            kld, fd = evaluate(batch_eval_x, ebflow, score_oracle, prob_oracle)
            klds.append(kld)
            fds.append(fd)
        
        fd_mean, fd_ci = mean_confidence_interval(fds)
        kld_mean, kld_ci = mean_confidence_interval(klds)
        print("="*10)
        print("results_toy/"+dataset+"/"+loss)
        print("fd: {:.2e} +- {:.2e}".format(fd_mean, fd_ci))
        print("kld: {:.2e} +- {:.2e}".format(kld_mean, kld_ci))
        print("="*10)

if __name__ == "__main__":
    app.run(run_test)
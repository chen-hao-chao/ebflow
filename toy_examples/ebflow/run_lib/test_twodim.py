import torch
import numpy as np
from absl import app

from ebflow.datasets.datasets import get_dataset
from ebflow.models import coupling_net
from ebflow.models.ema import ExponentialMovingAverage
from ebflow.run_lib.utils import get_oracle_score, evaluate, plot_energy

def run_test(config):
    restore_path = config['restore']

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
        
    checkpoint_file = restore_path
    checkpoint = torch.load(checkpoint_file, map_location='cuda')
    ebflow.load_state_dict(checkpoint['model'])
    ema.load_state_dict(checkpoint['ema'])
    print("Load checkpoint "+checkpoint_file)
    ema.copy_to(ebflow.parameters())
    kld, fd = evaluate(batch_eval_x, ebflow, score_oracle, prob_oracle)
    
    file_name = "results/energy.png"
    plot_energy(config, ebflow, file_name)
    
    print("="*10)
    print("fd: {:.3e}".format(fd.mean()))
    print("kld: {:.3e}".format(kld.mean()))
    print("="*10)

if __name__ == "__main__":
    app.run(run_test)
import os
import numpy as np
import torch
from torch.utils import tensorboard
import datetime

from ebflow.datasets.datasets import get_dataset
from ebflow.models import coupling_net
from ebflow.models.ema import ExponentialMovingAverage
from ebflow.run_lib.utils import get_oracle_score, evaluate, plot_energy, save_checkpoint

def run(config):
    # ==========================
    # Directory and Log
    # ==========================
    # Create directories for experimental logs.
    resultdir = config['resultdir']
    workdir = os.path.join(resultdir, config['workdir'])
    visualization_dir = os.path.join(workdir, "visualization")
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    tb_dir = os.path.join(workdir, "tensorboard")
    loc_dt_format = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
    tb_current_file_dir = os.path.join(tb_dir, loc_dt_format)
    try:
        os.makedirs(resultdir, exist_ok = True)
        os.makedirs(workdir, exist_ok = True)
        os.makedirs(visualization_dir, exist_ok = True)
        os.makedirs(checkpoint_dir, exist_ok = True)
        os.makedirs(tb_dir, exist_ok = True)
        os.makedirs(tb_current_file_dir, exist_ok = True)
        print("Directory created successfully")
    except OSError as error:
        print("Directory can not be created")

    # Tensorboard
    writer = tensorboard.SummaryWriter(tb_current_file_dir)

    # ==========================
    # Training State Init
    # ==========================
    # Model Setups
    ebflow = coupling_net.Net(n_blocks=config['n_blocks'],
                              hid=config['hidden_size'],
                              input_size=config['input_size'],
                              weight_init=config['weight_init'],
                              t_net_act=config['t_net_act'],
                              dependency=bool(config['dependency']),
                              norm_type=config['norm_type'],
                              sigma=config['sigma']).to('cuda', dtype=torch.double)
    # Optimizer
    if config['opt_type'] == "adam":
        optimizer = torch.optim.Adam(ebflow.parameters(),
                                    lr=config['lr'],
                                    weight_decay=config['weight_decay'],
                                    betas=(0.9, 0.999))
    elif config['opt_type'] == "adamW":
        optimizer = torch.optim.AdamW(ebflow.parameters(),
                                    lr=config['lr'],
                                    weight_decay=config['weight_decay'],
                                    betas=(0.9, 0.999))
    else:
        raise ValueError("Optimizor {} not recognized.".format(config['opt_type']))
                                    
    # Exponential Moving Average (EMA)
    ema = ExponentialMovingAverage(ebflow.parameters(), decay=0.999)

    # Final State
    state = dict(optimizer=optimizer, model=ebflow, ema=ema, step=0)

    # ==========================
    # Training and Eval Setups
    # ==========================
    # Build data iterators
    train_ds, test_ds = get_dataset(config)
    train_iter = iter(train_ds)
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

    # ==========================
    # Training Iterations
    # ==========================
    for t in range(config['n_iters']):
        noise_scale = config['std']
        # Get data
        data = next(train_iter)
        batch = torch.from_numpy(data['position']._numpy()).to('cuda', dtype=torch.double)
        
        std = torch.empty(batch.shape, device=batch.device).fill_(noise_scale)
        noise = torch.randn_like(batch, device=batch.device)
        x = torch.tensor(batch + std * noise, requires_grad=True).to('cuda', dtype=torch.double) 
        
        # Loss Function
        optimizer.zero_grad()
        ebflow.train().double()
        
        if config['loss_type'] == 'ml':
            log_p = ebflow.log_p(x)
            loss = (-log_p).mean()
        elif config['loss_type'] == 'sml':
            batch_size = x.shape[0]
            with torch.no_grad():
                z = torch.randn(x.shape).to(x.device)
                x_fake = ebflow.inverse(z)
                x = torch.cat([x, x_fake], 0)
            neg_e, _, _ = ebflow.neg_energy(x)
            energy_true = -neg_e[:batch_size]
            energy_fake = -neg_e[batch_size:]
            loss = energy_true - energy_fake
            loss = loss.mean()
        elif config['loss_type'] == 'dsm':
            neg_e, _, _ = ebflow.neg_energy(x)
            score = torch.autograd.grad(neg_e.sum(), x, create_graph=True)[0]
            loss = torch.sum(torch.square(score + noise/std), dim=1) * 0.5
            loss = loss.mean()
        elif config['loss_type'] == 'ssm':
            # code from https://github.com/ermongroup/sliced_score_matching
            neg_e, _, _ = ebflow.neg_energy(x)
            v = torch.randn_like(x, device=x.device).sign()
            score = torch.autograd.grad(neg_e.sum(), x, create_graph=True)[0]
            square = 0.5*torch.sum(score**2, dim=1)
            vs = torch.sum(score * v, dim=1)
            gvs = torch.autograd.grad(torch.sum(vs), x, create_graph=True)[0]
            trace = torch.sum(v*gvs, dim=1) 
            loss = (square+trace).mean()
        elif config['loss_type'] == 'fdssm':
            # code from https://github.com/taufikxu/FD-ScoreMatching
            neg_e, _, _ = ebflow.neg_energy(x)
            eps = 0.1
            dim = 2
            v = torch.randn_like(x, device=x.device)
            v_norm = torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True)+1e-8)
            v = v / v_norm * eps
            batch_size = x.shape[0]
            cat_input = torch.cat([x, x + v, x - v], 0)
            cat_output, _, _ = ebflow.neg_energy(cat_input)
            out_1 = -cat_output[:batch_size]
            out_2 = -cat_output[batch_size : 2 * batch_size]
            out_3 = -cat_output[2 * batch_size :]
            diffs_1 = out_2 - out_3
            loss1 = (diffs_1 ** 2) / 8
            loss2 = -out_2 - out_3 + 2 * out_1
            loss = (loss1 + loss2).mean() / (eps ** 2) * dim
        else:
            raise ValueError("Loss {} not recognized.".format(config.setup.loss_type))

        if not torch.isnan(loss).any():
            loss.backward()
            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(ebflow.parameters(), max_norm=config['grad_clip'])
            optimizer.step()
            state['step'] += 1
            state['ema'].update(ebflow.parameters())
        else:
            print("Loss {} is nan at step {}.".format(loss, t))

        if t % config['log_freq'] == 0:
            if bool(config['use_ema']):
                ema.store(ebflow.parameters())
                ema.copy_to(ebflow.parameters())
            
            # Evaluation
            kld, fd = evaluate(batch_eval_x, ebflow, score_oracle, prob_oracle)
            # Tensorboard
            writer.add_scalar("eval_fd", fd, t)
            writer.add_scalar("eval_kld", kld, t)
            writer.add_scalar("loss", loss, t)
            # Log info
            print('iter %s:' % t, 'eval_fd = %.3f' % fd)
            print('iter %s:' % t, 'eval_kld = %.3f' % kld)
            print('iter %s:' % t, 'training_loss = %.3f' % loss)
            print("----")
            if bool(config['use_ema']):
                ema.restore(ebflow.parameters())

        if t % config['plot_freq'] == 0:
            if bool(config['use_ema']):
                ema.store(ebflow.parameters())
                ema.copy_to(ebflow.parameters())
                plot_energy(config, ebflow, os.path.join(visualization_dir, str(t)+'_energy.png'))
            if bool(config['use_ema']):
                ema.restore(ebflow.parameters())
            
        if t % config['log_weight_freq'] == 0:
            save_checkpoint(os.path.join(checkpoint_dir, 'checkpoint_{}.pth'.format(t)), state)


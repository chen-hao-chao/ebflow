import os
import torch
from torch.utils import tensorboard
import datetime

from ebflow.datasets.datasets import get_dataset
from ebflow.models import linear_net

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
    ebflow = linear_net.Net(n_blocks=config['n_blocks'],
                              input_size=config['input_size'],
                              dependency=bool(config['dependency']),
                              type=config['mask_type'],
                              sigma=config['sigma']).to('cuda', dtype=torch.double)
    # Optimizer
    optimizer = torch.optim.Adam(ebflow.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'],
                                 betas=(0.9, 0.999))
                                
    # Final State
    state = dict(optimizer=optimizer, model=ebflow, step=0)

    # ==========================
    # Training and Eval Setups
    # ==========================
    # Build data iterators
    train_ds, test_ds = get_dataset(config)
    train_iter = iter(train_ds)
    test_iter = iter(test_ds)

    # Get an eval batch
    data = next(test_iter)
    batch_eval_x = torch.from_numpy(data['position']._numpy()).to('cuda', dtype=torch.double)

    # Evaluation
    ebflow.eval()
    nll = (-ebflow.log_p(batch_eval_x)).mean()
    # Tensorboard
    writer.add_scalar("eval_nll_loss", nll, 0)
    # Log info
    print('iter %s:' % 0, 'eval_nll_loss = %.3f' % nll)
    print("----")

    # ==========================
    # Training Iterations
    # ==========================
    for t in range(config['n_iters']):
        # Get data
        data = next(train_iter)
        batch = torch.from_numpy(data['position']._numpy()).to('cuda', dtype=torch.double)
        x = torch.tensor(batch, requires_grad=True).to('cuda', dtype=torch.double)
        
        # Loss Function
        optimizer.zero_grad()
        ebflow.train().double()
        log_p = ebflow.log_p(x)
        loss = (-log_p).mean()
       
        if not torch.isnan(loss).any():
            loss.backward()
            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(ebflow.parameters(), max_norm=config['grad_clip'])
            optimizer.step()
            state['step'] += 1
        else:
            print("Loss {} is nan at step {}.".format(loss, t))

        if t % config['log_freq'] == 0 and t != 0:
            # Evaluation
            ebflow.eval()
            nll = (-ebflow.log_p(batch_eval_x)).mean()
            # Tensorboard
            writer.add_scalar("eval_nll_loss", nll, t)
            writer.add_scalar("loss", loss, t)
            # Log info
            print('iter %s:' % t, 'eval_nll_loss = %.3f' % nll)
            print('iter %s:' % t, 'training_loss = %.3f' % loss)
            print("----")

        if t % config['log_weight_freq'] == 0:
            ckpt_dir = os.path.join(checkpoint_dir, 'checkpoint_{}.pth'.format(t))
            saved_state = {
                'optimizer': state['optimizer'].state_dict(),
                'model': state['model'].state_dict(),
                'step': state['step']
            }
            torch.save(saved_state, ckpt_dir)

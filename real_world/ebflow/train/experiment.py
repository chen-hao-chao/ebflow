# Code modified from https://github.com/akandykeller/SelfNormalizingFlows
import os
import torch
import torchvision

import datetime
from torch.utils import tensorboard

class Experiment:
    def __init__(self, model, trans, ema, train_loader, val_loader, test_loader,
                 optimizer, scheduler, datasize=(1,28,28), **kwargs):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.trans = trans
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.datasize = datasize

        try:
            self.data_shape = self.train_loader.dataset.dataset.data.shape[1:]
        except AttributeError:
            if type(train_loader.dataset.dataset) == torchvision.datasets.ImageFolder:
                self.data_shape = train_loader.dataset.dataset[0][0].shape
            else:
                self.data_shape = datasize

        # Useful conversion functions
        self.to_bpd = lambda x: x / (torch.log(torch.tensor(2.0)) 
                                     * torch.prod(torch.tensor(self.data_shape)))
        self.to_sliced_tensor = lambda x, k: x.unsqueeze(0).expand(k, *x.shape).contiguous().view(-1, x.shape[1], x.shape[2], x.shape[3])

        # Update config file
        self.config = {}
        self.config.update(**kwargs)

        # Update paths and create directories
        self.config['save_path'] = os.path.join(self.config['resultdir'], self.config['workdir'])        
        self.config['checkpoint_path'] = os.path.join(self.config['save_path'], "checkpoints")
        self.config['sample_dir'] = os.path.join(self.config['save_path'], "sample")
        loc_dt_format = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
        tb_dir = os.path.join(self.config['save_path'], "tensorboard")
        self.config['tb_path'] = os.path.join(tb_dir, loc_dt_format)
        try:
            os.makedirs(self.config['checkpoint_path'], exist_ok = True)
            os.makedirs(self.config['sample_dir'], exist_ok = True)
            os.makedirs(tb_dir, exist_ok = True)
            os.makedirs(self.config['tb_path'], exist_ok = True)
            print("Directory created successfully")
        except OSError as error:
            print("Directory can not be created")
        
        # Create initial entries for self.summary
        self.summary = {}
        self.update_summary('Epoch', 0)
        self.update_summary("Best Val LogPx", float('-inf'))
        self.update_summary("Test LogPx", float('-inf'))
        
        # Load checkpoint
        if self.config['restore_path']:
            self.load(self.config['restore_path'])

    def run(self):
        writer = tensorboard.SummaryWriter(self.config['tb_path'])
        
        # Evaluate before training
        # ------------
        val_logpx = self.eval_epoch(self.val_loader)
        val_logpx = val_logpx if val_logpx<500 else 500
        self.log('Val LogPx', val_logpx)
        self.log('Val BPD', self.to_bpd(val_logpx))
        writer.add_scalar("val_logp", val_logpx, 0)
        writer.add_scalar("val_logp_bpd", self.to_bpd(val_logpx), 0)

        if val_logpx > self.summary['Best Val LogPx']:
            self.update_summary('Best Val LogPx', val_logpx)
            self.update_summary('Best Val BPD', self.to_bpd(val_logpx))
            test_logpx = self.eval_epoch(self.test_loader)
            self.log('Test LogPx', test_logpx)
            self.log('Test BPD', self.to_bpd(test_logpx))
            self.update_summary('Test LogPx', test_logpx)
            self.update_summary('Test BPD', self.to_bpd(test_logpx))
            writer.add_scalar("test_logp", test_logpx, 0)
            writer.add_scalar("test_logp_bpd", self.to_bpd(test_logpx), 0)
        # ------------

        for e in range(self.summary['Epoch'] + 1, self.config['epochs'] + 1):
            self.update_summary('Epoch', e)
            print("="*5)
            print("Epoch: {}".format(e))

            # Perform training step
            avg_loss = self.train_epoch(e)
            self.log('Train Avg Loss', avg_loss)
            writer.add_scalar("avg_loss", avg_loss, e)

            # Perform evaluation
            if e % self.config['eval_epochs'] == 0:
                if self.ema is not None:
                    self.ema.store(self.model.parameters())
                    self.ema.copy_to(self.model.parameters())
                
                # Calculate gradient norm
                weights = None
                for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
                    weights = torch.cat((weights, p.grad.flatten()), 0) if weights is not None else p.grad.flatten()
                norm = torch.sqrt((weights**2).sum())   
                
                # Log information and save checkpoints
                val_logpx = self.eval_epoch(self.val_loader)
                self.log('Val LogPx', val_logpx)
                self.log('Val BPD', self.to_bpd(val_logpx))
                self.log('Gradient Norm', norm)
                writer.add_scalar("val_logp", val_logpx, e)
                writer.add_scalar("val_logp_bpd", self.to_bpd(val_logpx), e)
                writer.add_scalar("grad_norm", norm, e)

                if val_logpx > self.summary['Best Val LogPx']:
                    self.update_summary('Best Val LogPx', val_logpx)
                    self.update_summary('Best Val BPD', self.to_bpd(val_logpx))

                    # Evaluate using testing set
                    test_logpx = self.eval_epoch(self.test_loader)
                    self.log('Test LogPx', test_logpx)
                    self.log('Test BPD', self.to_bpd(test_logpx))
                    self.update_summary('Test LogPx', test_logpx)
                    self.update_summary('Test BPD', self.to_bpd(test_logpx))
                    writer.add_scalar("test_logp", test_logpx, e)
                    writer.add_scalar("test_logp_bpd", self.to_bpd(test_logpx), e)

                    # Checkpoint model
                    self.save(mode='best')
                
                # Checkpoint model
                self.save(mode='cur')

                if self.ema is not None:
                    self.ema.restore(self.model.parameters())

            # Visualize the generated images
            if e % self.config['sample_epochs'] == 0:
                if self.ema is not None:
                    self.ema.store(self.model.parameters())
                    self.ema.copy_to(self.model.parameters())
                self.sample(e)
                if self.ema is not None:
                    self.ema.restore(self.model.parameters())

    def log(self, name, val):
        print("{}: {}".format(name, val))

    def update_summary(self, name, val):
        self.summary[name] = val

    def get_loss(self, x):
        with torch.no_grad():
            x, _ = self.trans.forward(x)
            dim = torch.flatten(x, start_dim=1).shape[1]

        if self.config['loss'] == 'ml':
            lossval = -self.model.log_prob(x, zero_ldj=False)  
            lossval[lossval != lossval] = 0.0
            lossval = (lossval).sum() / len(x)
        elif self.config['loss'] == 'sml':
            batch_size = x.shape[0]
            with torch.no_grad():
                x_sample = self.model.sample(x.shape[0])
                x = torch.cat([x, x_sample.view(x.shape)], 0)
            out, neg_e = self.model(x, zero_ldj=True)
            energy_true = -neg_e[:batch_size] # positive samples
            energy_fake = -neg_e[batch_size:] # negative samples
            lossval = energy_true - energy_fake
            lossval[lossval != lossval] = 0.0
            lossval = (lossval).sum() / len(x)
        elif self.config['loss'] == 'dsm':
            std = torch.empty(x.shape, device=x.device).fill_(self.config['std'])
            noise = torch.randn_like(x, device=x.device)
            xp = x + std * noise
            xp = xp.requires_grad_()
            out, neg_e = self.model(xp, zero_ldj=True)
            score = torch.autograd.grad(neg_e.sum(), xp, create_graph=True)[0]
            if self.config['lambda'] != 0:
                # Encourage low energy
                # See Section A2.2 of supplementary material
                reg = (-neg_e).mean()
                lossval = torch.sum(torch.square(score + noise/std) * 0.5, dim=1) + reg * self.config['lambda']
            else:
                lossval = torch.sum(torch.square(score + noise/std) * 0.5, dim=1)
            lossval[lossval != lossval] = 0.0
            lossval = (lossval).sum() / len(x) / dim
        elif self.config['loss'] == 'ssm':
            # code from https://github.com/ermongroup/sliced_score_matching
            x = self.to_sliced_tensor(x, self.config['slices'])
            v = torch.randn_like(x, device=x.device).sign()
            x = x.requires_grad_()
            out, neg_e = self.model(x, zero_ldj=True)
            score = torch.autograd.grad(neg_e.sum(), x, create_graph=True)[0]
            square = 0.5*torch.sum(score**2, dim=1)
            vs = torch.sum(score * v, dim=1)
            gvs = torch.autograd.grad(torch.sum(vs), x, create_graph=True)[0]
            trace = torch.sum(v*gvs, dim=1)
            if self.config['lambda'] != 0:
                # Encourage low energy
                # See Section A2.2 of supplementary material
                reg = (-neg_e).mean()
                lossval = (square+trace).view(self.config['slices'], -1).mean(dim=0) + reg * self.config['lambda']
            else:
                lossval = (square+trace).view(self.config['slices'], -1).mean(dim=0)
            lossval[lossval != lossval] = 0.0
            lossval = (lossval).sum() / len(x) / dim
        elif self.config['loss'] == 'fdssm':
            # code from https://github.com/taufikxu/FD-ScoreMatching
            eps = self.config['eps']
            v = torch.randn_like(x, device=x.device)
            v_norm = torch.sqrt(torch.sum(v ** 2, dim=(1,2,3), keepdim=True))
            v = v / v_norm * eps
            batch_size = x.shape[0]
            cat_input = torch.cat([x, x + v, x - v], 0)
            out, neg_e = self.model(cat_input, zero_ldj=True)
            out_1 = -neg_e[:batch_size]
            out_2 = -neg_e[batch_size : 2 * batch_size]
            out_3 = -neg_e[2 * batch_size :]
            diffs_1 = out_2 - out_3
            loss1 = (diffs_1 ** 2) / 8
            loss2 = -out_2 - out_3 + 2 * out_1
            if self.config['lambda'] != 0:
                # Encourage low energy
                # See Section A2.2 of supplementary material
                reg = (out_1).mean()
                lossval = loss1 + loss2 + reg * self.config['lambda']
            else:
                lossval = loss1 + loss2
            lossval[lossval != lossval] = 0.0
            lossval = (lossval).sum() / len(x) / (eps ** 2)
        
        return lossval

    def warmup_lr(self, epoch, num_batches):
        if epoch <= self.config['warmup_epochs']:
            for param_group in self.optimizer.param_groups:
                s = (((num_batches+1) + (epoch-1) * len(self.train_loader)) 
                        / (self.config['warmup_epochs'] * len(self.train_loader)))
                param_group['lr'] = self.config['lr'] * s

    def train_epoch(self, epoch):
        total_loss = 0
        num_batches = 0

        self.model.train()
        for x, _ in self.train_loader:
            if self.config['warmup_epochs'] > 0:
                self.warmup_lr(epoch, num_batches)
            self.optimizer.zero_grad()
            x = x.float().to('cuda')
            lossval = self.get_loss(x)
            lossval.backward()
            
            # Gradient clipping
            if self.config['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               self.config['grad_clip_norm'])
            # Update optimizer and EMA
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model.parameters())
            
            total_loss += lossval.item()
            num_batches += 1

            # Print the training loss for each mini-batch
            if num_batches % self.config['log_interval'] == 0:
                self.log('Train Batch Loss', lossval)

        self.scheduler.step()
        avg_loss = total_loss / num_batches
        return avg_loss

    def eval_epoch(self, dataloader):
        total_logpx = 0.0
        num_x = 0
        self.model.eval()
        for x, _ in dataloader:
            x = x.float().to('cuda')
            x, ldj = self.trans.forward(x)
            x.requires_grad_()
            logp = self.model.log_prob(x, zero_ldj=False)
            total_logpx += (logp + ldj).sum().detach().cpu().numpy()
            num_x += len(x)

        avg_logpx = total_logpx / num_x
        return avg_logpx

    def sample(self, e):
        n = self.config['n_samples']
        s_dir = self.config['sample_dir']
        s_path = os.path.join(s_dir, '{}.png'.format(e))

        with torch.no_grad():
            x_sample = self.model.sample(n)
            x_sample = self.trans.reverse(x_sample)

            if len(self.data_shape) == 2:
                x_sample = x_sample.view(n, 1, *self.data_shape)
            else:
                x_sample = x_sample.view(n, self.datasize[0], self.datasize[1], self.datasize[2])

        os.makedirs(s_dir, exist_ok=True)
        torchvision.utils.save_image(x_sample / 256.,
                                     s_path, nrow=10,
                                     padding=2, normalize=False)

    def save(self, mode='cur'):
        self.log('Note', 'Saving checkpoint to: {}'.format(self.config["checkpoint_path"]))
        if self.ema is not None:
            checkpoint = {'summary': self.summary,
                          'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(),
                          'scheduler_state_dict': self.scheduler.state_dict(),
                          'ema': self.ema.state_dict(),
                          'config': self.config}
        else:
            checkpoint = {'summary': self.summary,
                          'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(),
                          'scheduler_state_dict': self.scheduler.state_dict(),
                          'config': self.config}

        torch.save(checkpoint, os.path.join(self.config['checkpoint_path'], "checkpoint_"+mode+".tar"))

    def load(self, path):
        # Load models
        self.log('Note', 'Loading checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.summary = checkpoint['summary']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
            self.ema.copy_to(self.model.parameters())
        print("load models...")

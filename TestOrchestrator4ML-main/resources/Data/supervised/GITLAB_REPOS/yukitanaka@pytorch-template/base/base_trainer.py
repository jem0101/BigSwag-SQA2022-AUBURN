import json
import logging
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim

from utils import util


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.name = config['name']
        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.model = util.wrap_parallel_gpu(self.model, self.logger)
        self.device = torch.device("cuda:" + str(config['gpu']) if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.train_logger = train_logger
        self.optimizer = getattr(optim, config['optimizer_type'])(self.model.parameters(),
                                                                  **config['optimizer'])
        self.lr_scheduler = getattr(
            optim.lr_scheduler,
            config['lr_scheduler_type'], None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config['lr_scheduler'])
            self.lr_scheduler_freq = config['lr_scheduler_freq']
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1
        if resume:
            self._resume_checkpoint(resume)
        else:
            since = datetime.now().strftime('_%Y%m%d_%H_%M_%S')
            self.logger.info('make directory at {}'.format(since))
            self.checkpoint_dir = Path(config['trainer']['save_dir']) / Path(self.name + since)
            self.checkpoint_dir.mkdir(exist_ok=False)
            with (self.checkpoint_dir / 'config.json').open('w') as jn:
                json.dump(config, jn, indent=4, sort_keys=False)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics):
                        log[metric.__name__] = result['metrics'][i]
                elif key == 'val_metrics':
                    for i, metric in enumerate(self.metrics):
                        log['val_' + metric.__name__] = result['val_metrics'][i]
                else:
                    log[key] = value
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(key, value))
            if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) \
                    or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                self.monitor_best = log[self.monitor]
                self._save_checkpoint(epoch, log, save_best=True)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, log)
            if self.lr_scheduler and epoch % self.lr_scheduler_freq == 0:
                self.lr_scheduler.step(epoch)
                lr = self.lr_scheduler.get_lr()[0]
                self.logger.info('New Learning Rate: {:.6f}'.format(lr))
        self._plot_learning()
        self._save_history()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, log, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = Path(self.checkpoint_dir) / 'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'.format(epoch, log['loss'])
        torch.save(state, filename.as_posix())
        if save_best:
            filename.replace(self.checkpoint_dir / 'model_best.pth.tar')
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.checkpoint_dir = Path(resume_path).parent
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        self.train_logger = checkpoint['logger']
        self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def _plot_learning(self):
        """
        Plot loss and metrics
        """
        history = pd.read_json(str(self.train_logger))
        figs, ax = plt.subplots(len(self.metrics) + 1, 1, sharex='all', figsize=(9, 6))
        for key in history.index.values:
            if 'loss' in key:
                ax[0].plot(history.loc['epoch'], history.loc[key], 'o-', label=key)
                ax[0].set(ylabel='loss')
            for i, metric in enumerate(self.metrics):
                if key == metric.__name__:
                    ax[i + 1].plot(history.loc['epoch'], history.loc[key], 'o-', label=key)
                    ax[i + 1].plot(history.loc['epoch'], history.loc['val_' + key], 'o-', label='val_' + key)
                    ax[i + 1].set(ylabel='score')
        for a in ax:
            a.legend()
            a.grid()
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.savefig((self.checkpoint_dir / 'learning.png').as_posix())
        plt.close()

    def _save_history(self):
        pd.read_json(str(self.train_logger)).to_csv(self.checkpoint_dir / 'history.csv')

import torch
import numpy as np
from torchvision.utils import make_grid

from pytorch_3T27T.base import BaseTrainer, AverageMeter
from pytorch_3T27T.utils import setup_logger


logger = setup_logger(__name__)


__all__ = ['AlphaTrainer']


class AlphaTrainer(BaseTrainer):
    """
    Responsible for training loop and validation.
    """
    def __init__(self, model, loss, metrics, optimizer, start_epoch, config,
                 device, dataloader, val_dataloader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, start_epoch, config,
                         device, dataloader, val_dataloader, lr_scheduler)
        self.do_validation = self.val_dataloader is not None
        self.log_step = int(np.sqrt(self.dataloader.batch_size)) * 8


    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch.

        Returns
        -------
        Returns a dictionary with the results of this run, like the value for
        each metric, etc
        """
        self.model.train()

        loss_mtr = AverageMeter('loss')
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]

        for batch_idx, (data, target) in enumerate(self.dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            loss_mtr.update(loss.item(), data.size(0))

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch) * len(self.dataloader) +
                                     batch_idx)
                self.writer.add_scalar('batch/loss', loss.item())
                for mtr, value in zip(metric_mtrs,
                                      self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))
                    self.writer.add_scalar(f'batch/{mtr.name}', value)
                self._log_batch(epoch, batch_idx, self.dataloader.batch_size,
                                len(self.dataloader), loss.item())

            if batch_idx == 0:
                self.writer.add_image('data', make_grid(data.cpu(), nrow=8,
                                                        normalize=True))
        del data
        del target
        del output
        torch.cuda.empty_cache()

        self.writer.add_scalar('epoch/loss', loss_mtr.avg)
        for mtr in metric_mtrs:
            self.writer.add_scalar(f'epoch/{mtr.name}', mtr.avg)

        results = {
            'loss': loss_mtr.avg,
            'metrics': [mtr.avg for mtr in metric_mtrs]
        }

        if self.do_validation:
            val_results = self._val_epoch(epoch)
            results = {**results, **val_results}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results


    def _log_batch(self, epoch, batch_idx, batch_size, len_data, loss):
        n_samples = batch_size * len_data
        n_complete = batch_idx * batch_size
        percent = 100.0 * batch_idx / len_data
        msg = (f'Train Epoch: {epoch} [{n_complete}/{n_samples} '
               f'({percent:.0f}%)] Loss: {loss:.6f}')
        logger.debug(msg)


    def _eval_metrics(self, output, target):
        with torch.no_grad():
            for metric in self.metrics:
                value = metric(output, target)
                yield value


    def _val_epoch(self, epoch: int) -> dict:
        """
        Validate after training epoch.

        Returns
        -------
        Returns a dictionary with the keys 'val_loss' and 'val_metrics'
        """
        self.model.eval()
        loss_mtr = AverageMeter('loss')
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target)
                loss_mtr.update(loss.item(), data.size(0))
                for mtr, value in zip(metric_mtrs,
                                      self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))
                if batch_idx == 0:
                    self.writer.add_image('input', make_grid(data.cpu(),
                                                             nrow=8,
                                                             normalize=True))

        del data
        del target
        del output
        torch.cuda.empty_cache()

        self.writer.set_step(epoch, 'val')
        self.writer.add_scalar('loss', loss_mtr.avg)
        for mtr in metric_mtrs:
            self.writer.add_scalar(mtr.name, mtr.avg)

        return {
            'val_loss': loss_mtr.avg,
            'val_metrics': [mtr.avg for mtr in metric_mtrs]
        }

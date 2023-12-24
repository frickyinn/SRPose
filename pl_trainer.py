import numpy as np
import torch
import lightning as L
from lightglue import SuperPoint

from utils import rot_degree_error
from model import LightPose


class PL_LightPose(L.LightningModule):
    def __init__(
            self,
            task,
            lr,
            num_keypoints,
            steps_per_epoch,
            epochs,
            n_layers=3,
            num_heads=4,
            features='superpoint',
        ):
        super().__init__()
        
        self.extractor = SuperPoint(max_num_keypoints=num_keypoints, detection_threshold=0.0).eval()
        self.module = LightPose(features=features, task=task, n_layers=n_layers, num_heads=num_heads)
        self.criterion = torch.nn.HuberLoss()
        self.degree_errors = {k:[] for k in ['train', 'valid', 'test']}
        self.meter_errors = {k:[] for k in ['train', 'valid', 'test']}

        self.save_hyperparameters()

    def _shared_log(self, mode, loss, loss_r, loss_t, loss_t_scale, degrees, meters):
        degrees = degrees * 180 / torch.pi
        self.log_dict({
            f'{mode}_loss': loss,
            f'{mode}_degree_avg': degrees.mean(),
            f'{mode}_meter_avg': meters.mean(),
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log_dict({
            f'{mode}_loss_r': loss_r,
            f'{mode}_loss_t': loss_t,
            f'{mode}_loss_scale': loss_t_scale,
            f'{mode}_10d_acc': (degrees <= 10).float().mean(),
            f'{mode}_15d_acc': (degrees <= 15).float().mean(),
            f'{mode}_30d_acc': (degrees <= 30).float().mean(),
            f'{mode}_1m_acc': (meters <= 1).float().mean(),
        }, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        images = batch['images']
        rotation = batch['rotation']
        translation = batch['translation']
        intrinsics = batch['intrinsics']

        image0 = images[:, 0, ...]
        image1 = images[:, 1, ...]

        with torch.no_grad():
            feats0 = self.extractor({'image': image0})
            feats1 = self.extractor({'image': image1})

        if 'scales' in batch:
            scales = batch['scales']
            feats0['keypoints'] *= scales[:, 0].unsqueeze(1)
            feats1['keypoints'] *= scales[:, 1].unsqueeze(1)

        if self.hparams.task == 'scene':
            pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': intrinsics[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})
        elif self.hparams.task == 'object':
            bboxes = batch['bboxes']
            pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': intrinsics[:, 0], 'bbox': bboxes[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})

        err_r = rot_degree_error(pred_r, rotation)
        loss_r = self.criterion(err_r, torch.zeros_like(err_r))

        loss_t = self.criterion(pred_t, translation)
        loss_t_scale = self.criterion(pred_t / pred_t.norm(2, dim=1, keepdim=True), translation / translation.norm(2, dim=1, keepdim=True))

        loss = loss_r + loss_t + loss_t_scale

        degrees = err_r.detach()
        meters = (pred_t.detach() - translation).norm(2, dim=1)

        self.degree_errors['train'].append(degrees)
        self.meter_errors['train'].append(meters)

        self._shared_log('train', loss, loss_r, loss_t, loss_t_scale, degrees, meters)

        return loss
    
    def _shared_eval_step(self, batch, batch_idx):
        images = batch['images']
        rotation = batch['rotation']
        translation = batch['translation']
        intrinsics = batch['intrinsics']

        image0 = images[:, 0, ...]
        image1 = images[:, 1, ...]

        with torch.no_grad():
            feats0 = self.extractor({'image': image0})
            feats1 = self.extractor({'image': image1})

        if 'scales' in batch:
            scales = batch['scales']
            feats0['keypoints'] *= scales[:, 0].unsqueeze(1)
            feats1['keypoints'] *= scales[:, 1].unsqueeze(1)

        if self.hparams.task == 'scene':
            pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': intrinsics[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})
        elif self.hparams.task == 'object':
            bboxes = batch['bboxes']
            pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': intrinsics[:, 0], 'bbox': bboxes[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})

        err_r = rot_degree_error(pred_r, rotation)
        loss_r = self.criterion(err_r, torch.zeros_like(err_r))

        loss_t = self.criterion(pred_t, translation)
        loss_t_scale = self.criterion(pred_t / pred_t.norm(2, dim=1, keepdim=True), translation / translation.norm(2, dim=1, keepdim=True))

        loss = loss_r + loss_t + loss_t_scale

        degrees = err_r.detach()
        meters = (pred_t.detach() - translation).norm(2, dim=1)

        return loss, loss_r, loss_t, loss_t_scale, degrees, meters

    def validation_step(self, batch, batch_idx):
        loss, loss_r, loss_t, loss_t_scale, degrees, meters = self._shared_eval_step(batch, batch_idx)

        self.degree_errors['valid'].append(degrees)
        self.meter_errors['valid'].append(meters)

        self._shared_log('valid', loss, loss_r, loss_t, loss_t_scale, degrees, meters)

    def test_step(self, batch, batch_idx):
        loss, loss_r, loss_t, loss_t_scale, degrees, meters = self._shared_eval_step(batch, batch_idx)

        self.degree_errors['test'].append(degrees)
        self.meter_errors['test'].append(meters)

        self._shared_log('test', loss, loss_r, loss_t, loss_t_scale, degrees, meters)

    def _shared_on_epoch_end(self, mode):
        degree_erros = torch.hstack(self.degree_errors[mode]) * 180 / torch.pi
        degree_auc = error_auc(degree_erros.cpu(), [5, 10, 20], mode)
        degree_med = degree_erros.median()
        meter_med = torch.hstack(self.meter_errors[mode]).median()

        self.log_dict({
            **degree_auc,
            f'{mode}_degree_med': degree_med,
            f'{mode}_meter_med': meter_med,
        }, sync_dist=True)

        self.degree_errors[mode].clear()
        self.meter_errors[mode].clear()

    def on_train_epoch_end(self):
        self._shared_on_epoch_end('train')

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end('valid')

    def on_test_epoch_end(self):
        self._shared_on_epoch_end('test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=self.hparams.steps_per_epoch, epochs=self.hparams.epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

def error_auc(errors, thresholds, mode):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'{mode}_degree_auc@{t}': auc for t, auc in zip(thresholds, aucs)}

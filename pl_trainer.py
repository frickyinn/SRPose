from typing import Any
import numpy as np
import torch
import lightning as L
from lightglue import SuperPoint

from utils import rotation_angular_error, translation_angular_error, error_auc
from model import LightPose


class PL_LightPose(L.LightningModule):
    def __init__(
            self,
            task,
            lr,
            epochs,
            pct_start,
            num_keypoints,
            n_layers,
            num_heads,
            features='superpoint',
        ):
        super().__init__()
        
        self.extractor = SuperPoint(max_num_keypoints=num_keypoints, detection_threshold=0.0).eval()
        self.module = LightPose(features=features, task=task, n_layers=n_layers, num_heads=num_heads)
        self.criterion = torch.nn.HuberLoss()

        self.s_r = torch.nn.Parameter(torch.zeros(1))
        # self.s_ta = torch.nn.Parameter(torch. zeros(1))
        self.s_t = torch.nn.Parameter(torch.zeros(1))

        self.r_errors = {k:[] for k in ['train', 'valid', 'test']}
        self.ta_errors = {k:[] for k in ['train', 'valid', 'test']}
        self.t_errors = {k:[] for k in ['train', 'valid', 'test']}

        self.save_hyperparameters()

    def _shared_log(self, mode, loss, loss_r, loss_t, loss_ta, loss_tn):
        self.log_dict({
            f'{mode}_loss': loss,
            f'{mode}_loss_r': loss_r,
            f'{mode}_loss_t': loss_t,
            f'{mode}_loss_ta': loss_ta,
            f'{mode}_loss_tn': loss_tn,
        }, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss, loss_r, loss_ta, loss_t, loss_tn, r_err, ta_err, t_err = self._shared_forward_step(batch, batch_idx)

        self.r_errors['train'].append(r_err)
        self.ta_errors['train'].append(ta_err)
        self.t_errors['train'].append(t_err)

        self._shared_log('train', loss, loss_r, loss_t, loss_ta, loss_tn)
        # self.log('s_r', self.s_r)
        # self.log('s_t', self.s_t)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_r, loss_ta, loss_t, loss_tn, r_err, ta_err, t_err = self._shared_forward_step(batch, batch_idx)

        self.r_errors['valid'].append(r_err)
        self.ta_errors['valid'].append(ta_err)
        self.t_errors['valid'].append(t_err)

        self._shared_log('valid', loss, loss_r, loss_t, loss_ta, loss_tn)

    def test_step(self, batch, batch_idx):
        loss, loss_r, loss_ta, loss_t, loss_tn, r_err, ta_err, t_err = self._shared_forward_step(batch, batch_idx)

        self.r_errors['test'].append(r_err)
        self.ta_errors['test'].append(ta_err)
        self.t_errors['test'].append(t_err)

        self._shared_log('test', loss, loss_r, loss_t, loss_ta, loss_tn)
    
    def _shared_forward_step(self, batch, batch_idx):
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

        r_err = rotation_angular_error(pred_r, rotation)
        ta_err = translation_angular_error(pred_t, translation)

        loss_r = self.criterion(r_err, torch.zeros_like(r_err))
        loss_ta = self.criterion(ta_err, torch.zeros_like(ta_err))
        loss_tn = self.criterion(pred_t / pred_t.norm(2, dim=-1, keepdim=True), translation / translation.norm(2, dim=-1, keepdim=True))
        loss_t = self.criterion(pred_t, translation)

        # loss = loss_r * torch.exp(-self.s_r) + loss_t * torch.exp(-self.s_t) + loss_ta * torch.exp(-self.s_ta) + self.s_r + self.s_t + self.s_ta
        loss = loss_r + loss_ta + loss_t + loss_tn

        r_err = r_err.detach()
        ta_err = ta_err.detach()
        t_err = (pred_t.detach() - translation).norm(2, dim=1)

        return loss, loss_r, loss_ta, loss_t, loss_tn, r_err, ta_err, t_err

    def predict_one_data(self, data, device='cuda'):
        images = data['images'].to(device)
        intrinsics = data['intrinsics'].to(device)

        image0 = images[:, 0, ...]
        image1 = images[:, 1, ...]

        with torch.no_grad():
            feats0 = self.extractor({'image': image0})
            feats1 = self.extractor({'image': image1})

        if 'scales' in data:
            scales = data['scales'].to(device)
            feats0['keypoints'] *= scales[:, 0].unsqueeze(1)
            feats1['keypoints'] *= scales[:, 1].unsqueeze(1)

        if self.hparams.task == 'scene':
            pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': intrinsics[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})
        elif self.hparams.task == 'object':
            bboxes = data['bboxes'].to(device)
            pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': intrinsics[:, 0], 'bbox': bboxes[:, 0]}, 'image1': {**feats1, 'intrinsics': intrinsics[:, 1]}})

        return pred_r[0], pred_t[0]

    def _shared_on_epoch_end(self, mode):
        r_errors = torch.hstack(self.r_errors[mode]).rad2deg()
        ta_errors = torch.hstack(self.ta_errors[mode]).rad2deg()
        ta_errors = torch.minimum(ta_errors, 180-ta_errors)
        
        auc = error_auc(torch.maximum(r_errors, ta_errors).cpu(), [5, 10, 20], mode)
        t_errors = torch.hstack(self.t_errors[mode])

        self.log_dict({
            **auc,
            f'{mode}_r_avg': r_errors.mean(),
            f'{mode}_r_med': r_errors.median(),
            f'{mode}_r_30d': (r_errors < 30).float().mean(),
            f'{mode}_r_15d': (r_errors < 15).float().mean(),
            f'{mode}_ta_avg': ta_errors.mean(),
            f'{mode}_ta_med': ta_errors.median(),
            f'{mode}_t_avg': t_errors.mean(),
            f'{mode}_t_med': t_errors.median(),
            f'{mode}_t_10cm': (t_errors < 0.1).float().mean(),
        }, sync_dist=True)

        self.r_errors[mode].clear()
        self.ta_errors[mode].clear()
        self.t_errors[mode].clear()

    def on_train_epoch_end(self):
        self._shared_on_epoch_end('train')

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end('valid')

    def on_test_epoch_end(self):
        self._shared_on_epoch_end('test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=1, epochs=self.hparams.epochs, pct_start=self.hparams.pct_start)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

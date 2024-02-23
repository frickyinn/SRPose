import torch

from .relpose import RelPose
from .pl_trainer import PL_RelPose, keypoint_dict


class SparseRelPose():
    def __init__(self, ckpt_path, num_keypoints=2048, device='cuda'):
        ckpt = torch.load(ckpt_path)
        hparams = ckpt['hparams']

        self.extractor = keypoint_dict[hparams['features']](max_num_keypoints=num_keypoints, detection_threshold=0.0).eval().to(device)
        self.module = RelPose(
            features=hparams['features'],
            task=hparams['task'], 
            n_layers=hparams['n_layers'], 
            num_heads=hparams['num_heads']
        ).eval().to(device)
        self.module.load_state_dict(ckpt['state_dict'])
        
        self.task = hparams['task']
        self.device = device
    
    @torch.no_grad()
    def regress(self, image0, image1, K0, K1, bbox=None, scales=None):
        image0 = image0.to(self.device)[None]
        image1 = image1.to(self.device)[None]

        K0, K1 = K0.to(self.device)[None], K1.to(self.device)[None]

        feats0 = self.extractor({'image': image0})
        feats1 = self.extractor({'image': image1})

        if scales is not None:
            scales = scales.to(self.device)[None]
            feats0['keypoints'] *= scales[:, 0].unsqueeze(1)
            feats1['keypoints'] *= scales[:, 1].unsqueeze(1)

        if self.task == 'scene':
            pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': K0}, 'image1': {**feats1, 'intrinsics': K1}})

        elif self.task == 'object':
            bbox = bbox.to(self.device)[None]
            pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': K0, 'bbox': bbox}, 'image1': {**feats1, 'intrinsics': K1}})

        return pred_r[0].cpu(), pred_t[0].cpu()

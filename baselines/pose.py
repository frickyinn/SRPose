import torch
from torchvision.transforms import Resize

from .matchers import LightGlue, LoFTR
# from .__models import SuperGlue, SGMNet, ASpanFormer, DKM
from .pose_solver import EssentialMatrixSolver, EssentialMatrixMetricSolver, PnPSolver, ProcrustesSolver

import time


class PoseRecover():
    def __init__(self, matcher='lightglue', solver='procrustes', img_resize=None, device='cuda'):
        self.device = device

        if matcher == 'lightglue':
            self.matcher = LightGlue(device=device)
        elif matcher == 'loftr':
            self.matcher = LoFTR(device=device)
        # elif matcher == 'superglue':
        #     self.matcher = SuperGlue(device=device)
        # elif matcher == 'aspanformer':
        #     self.matcher = ASpanFormer(device=device)
        # elif matcher == 'sgmnet':
        #     self.matcher = SGMNet(device=device)
        # elif matcher == 'dkm':
        #     self.matcher = DKM(device=device)
        else:
            raise NotImplementedError

        self.img_resize = img_resize

        self.basic_solver = EssentialMatrixSolver()

        if solver == 'essential':
            self.scaled_solver = EssentialMatrixMetricSolver()
        elif solver == 'pnp':
            self.scaled_solver = PnPSolver()
        elif solver == 'procrustes':
            self.scaled_solver = ProcrustesSolver()
        
    def recover(self, image0, image1, K0, K1, bbox0=None, bbox1=None, mask0=None, mask1=None, depth0=None, depth1=None):
        if self.img_resize is not None:
            h, w = image0.shape[-2:]
            if h > w:
                h_new = self.img_resize
                w_new = int(w * h_new / h)
            else:
                w_new = self.img_resize
                h_new = int(h * w_new / w)
                
            # h_new, w_new = 480, 640
            resize = Resize((h_new, w_new), antialias=True)
            scale0 = torch.tensor([image0.shape[-1]/w_new, image0.shape[-2]/h_new], dtype=torch.float)
            scale1 = torch.tensor([image1.shape[-1]/w_new, image1.shape[-2]/h_new], dtype=torch.float)
            image0 = resize(image0)
            image1 = resize(image1)

        points0, points1, preprocess_time, extract_time, match_time = self.matcher.match(image0, image1)

        if self.img_resize is not None:
            points0 *= scale0.unsqueeze(0).to(points0.device)
            points1 *= scale1.unsqueeze(0).to(points1.device)
        
        if bbox0 is not None and bbox1 is not None:
            x1, y1, x2, y2 = bbox0
            u1, v1, u2, v2 = bbox1

            points0[:, 0] += x1
            points0[:, 1] += y1

            points1[:, 0] += u1
            points1[:, 1] += v1

        if mask0 is not None and mask1 is not None:
            filtered_ind0 = mask0[(points0[:, 1]).int(), (points0[:, 0]).int()]
            filtered_ind1 = mask1[(points1[:, 1]).int(), (points1[:, 0]).int()]
            filtered_inds = filtered_ind0 * filtered_ind1
            points0 = points0[filtered_inds]
            points1 = points1[filtered_inds]

        points0, points1 = points0.cpu().numpy(), points1.cpu().numpy()

        start_time = time.time()

        if depth0 is None or depth1 is None:
            R_est, t_est, _ = self.basic_solver.estimate_pose(points0, points1, {'K_color0': K0, 'K_color1': K1})
        else:
            R_est, t_est, _ = self.scaled_solver.estimate_pose(points0, points1, {'K_color0': K0, 'K_color1': K1, 'depth0': depth0, 'depth1': depth1})

        recover_time = time.time()

        return R_est, t_est, points0, points1, preprocess_time, extract_time, match_time, recover_time-start_time

import os
import torch

from model import SparseRelPose
# from utils import project_3D_points, plot_3D_box


class PoseTracker():
    def __init__(self, K, model_type, pose_recover, device='cuda'):
        self.K = K
        self.device = device

        self.frame_count = 0
        self.pose_queue = []

        self.model_type = model_type
        self.pose_recover = pose_recover

    def run(self, rgb, depth, mask, bbox):
        # rgb: (3, H, W), in [0, 1]
        # depth: (H, W), in meters
        # mask: (H, W), in {0, 1}
        # bbox: (4,)

        rgb = rgb.to(self.device)
        depth = depth
        mask = mask.to(self.device)
        masked_rgb = rgb * mask

        if self.frame_count == 0:
            init_pose = torch.eye(4)
            self.pose_queue.append(init_pose)
            
            self.last_rgb = rgb
            self.last_masked_rgb = masked_rgb
            self.last_depth = depth
            self.last_bbox = bbox
            self.frame_count += 1
            return

        u1, v1, u2, v2 = self.last_bbox
        last_cropped_rgb = self.last_masked_rgb[:, v1:v2, u1:u2]
        
        x1, y1, x2, y2 = bbox
        cropped_rgb = masked_rgb[:, y1:y2, x1:x2]

        if self.model_type != 'relpose':
            R, t, points0, points1, io_time, ex_time, com_time, re_time = self.pose_recover.recover(last_cropped_rgb, cropped_rgb, self.K, self.K, self.last_bbox, bbox, None, None, self.last_depth, depth)
            R, t = torch.from_numpy(R), torch.from_numpy(t)
        else:
            R, t = self.pose_recover.regress(self.last_rgb, rgb, self.K, self.K, self.last_bbox)

        new_pose = torch.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t
        self.pose_queue.append(new_pose)

        self.last_rgb = rgb
        self.last_masked_rgb = masked_rgb
        self.last_depth = depth
        self.last_bbox = bbox
        self.frame_count += 1

    def get_last_pose(self):
        pose = self.pose_queue[-1]
        return pose[:3, :3], pose[:3, 3]

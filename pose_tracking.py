import os
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch
import torchvision
import pdb

from datasets.ho3d import HO3D
from RelPoseRepo.pose import PoseRecover
from model import LightPose as LightPose_
from lightglue import SuperPoint
from utils.metrics import add, adi, rotation_angular_error
from visualize import project_3D_points, plot_3D_box


class LightPose():
    def __init__(self, ckpt_path='checkpoints/ho3d.ckpt', device='cuda'):
        self.extractor = SuperPoint(max_num_keypoints=2048, detection_threshold=0.0).eval().to(device)
        self.module = LightPose_(features='superpoint', task='object', n_layers=6, num_heads=4).eval().to(device)
        self.module.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.device = device
    
    @torch.no_grad()
    def regress(self, image0, image1, K0, K1, bbox):
        image0 = image0.to(self.device)[None]
        image1 = image1.to(self.device)[None]

        K0, K1 = K0.to(self.device)[None], K1.to(self.device)[None]

        feats0 = self.extractor({'image': image0})
        feats1 = self.extractor({'image': image1})

        bbox = bbox.to(self.device)[None]
        pred_r, pred_t = self.module({'image0': {**feats0, 'intrinsics': K0, 'bbox': bbox}, 'image1': {**feats1, 'intrinsics': K1}})

        return pred_r[0].cpu(), pred_t[0].cpu()


def main(args):
    testset = HO3D(args.data_path, args.seq_path, mode='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    device = args.device
    img_resize = args.resize
    data_path = args.data_path

    if args.matcher != 'lightpose':
        poseRec = PoseRecover(matcher=args.matcher, solver=3, img_resize=img_resize, device=device)
    else:
        poseRec = LightPose()

    # R_gts, t_gts = [], []
    adds, adis, prjs = [], [], []
    frames = []
    
    objCorners = testset[0]['objCorners']
    image_f = testset[0]['color']
    proj_pts = project_3D_points(testset[0]['intrinsic'], objCorners)
    frame = plot_3D_box(proj_pts.int().numpy(), (image_f.permute(1, 2, 0) * 255.).numpy().astype(np.uint8), (0, 0, 255))
    frames.append(frame)

    mask_f = testset[0]['mask']
    depth_f = testset[0]['depth']
    bbox_f = testset[0]['bbox']

    for i, data in enumerate(tqdm(testloader)):
        if i == 0:
            continue
        image = data['color'][0]
        bbox = data['bbox'][0]

        x1, y1, x2, y2 = bbox
        image_bbox = image[:, y1:y2, x1:x2]
        u1, v1, u2, v2 = bbox_f
        image_f_bbox = image_f[:, v1:v2, u1:u2]

        mask = data['mask'][0]
        depth = data['depth'][0]

        K = data['intrinsic'][0]

        if args.matcher != 'lightpose':
            R, t, points0, points1, io_time, ex_time, com_time, re_time = poseRec.recover(image_f_bbox.to(device), image_bbox.to(device), K, K, bbox_f, bbox, mask_f.to(device), mask.to(device), depth_f, depth)
            R_pred, t_pred = torch.from_numpy(R), torch.from_numpy(t)
        else:
            R_pred, t_pred = poseRec.regress(image_f, image, K, K, bbox_f)
        
        # pdb.set_trace()

        proj_pts = project_3D_points(K, data['objCorners'][0])
        frame = plot_3D_box(proj_pts.int().numpy(), (image.permute(1, 2, 0) * 255.).numpy().astype(np.uint8), (0, 255, 0))

        objCorners = objCorners @ R_pred.mT.float() + t_pred.float()
        proj_pts_p = project_3D_points(K, objCorners)
        frame = plot_3D_box(proj_pts_p.int().numpy(), frame, (0, 0, 255))
        frames.append(frame)

        # if i == 100:
        #     print(R_pred)
        #     print(t_pred)
        #     plt.figure(figsize=(10, 5))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(frames[-2])
        #     # plt.scatter(points0[:, 0], points0[:, 1])
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(frames[-1])
        #     # plt.scatter(points1[:, 0], points1[:, 1])
        #     plt.show()
        #     break
    
        # pdb.set_trace()
        objCorners = data['objCorners'][0]
        image_f = image
        depth_f = depth
        mask_f = mask
        bbox_f = bbox


        # T = data['extrinsic'][0]
        # T = T.numpy()
        # R_gt = rotation_angular_error(torch.from_numpy(T[:3, :3])[None], torch.eye(3)[None])
        # R_gts.append(R_gt[0])
        # t_gt = torch.tensor(T[:3, 3]).norm(2)
        # t_gts.append(t_gt)

        # if np.isnan(R).any():
        #     adds.append(1.)
        #     adis.append(1.)
        #     prjs.append(40.)
        # else:
        #     adds.append(add(R, t, T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))
        #     adis.append(adi(R, t, T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))
        #     prjs.append(reproj(K1.numpy(), R, t, T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))

    # R_gts = torch.tensor(R_gts).rad2deg()
    # t_gts = torch.tensor(t_gts)
    # print(f'rel_rotation_avg:\t{R_gts.mean():.2f}')
    # print(f'rel_rotation_max:\t{R_gts.max():.2f}')
    # print(f'rel_translation_avg:\t{t_gts.mean():.2f}')
    # print(f'rel_translation_max:\t{t_gts.max():.2f}')

    # import pdb
    # pdb.set_trace()
    frames = torch.from_numpy(np.asarray(frames))
    output_path = os.path.join(args.output_path, f'{args.matcher}_{args.seq_path}.mp4')
    torchvision.io.write_video(output_path, frames, fps=30, video_codec="libx264")

    # print(f'ADD:\t\t{compute_continuous_auc(adds, np.linspace(0.0, 0.1, 1000)):.4f}')
    # print(f'ADD-S\t\t{compute_continuous_auc(adis, np.linspace(0.0, 0.1, 1000)):.4f}')
    # print(f'Proj.2D:\t{compute_continuous_auc(prjs, np.linspace(0.0, 40.0, 1000)):.4f}')


def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--task', type=str, help='scene | object', choices={'scene', 'object'}, required=True)
    # parser.add_argument('--dataset', type=str, help='matterport | megadepth | scannet | bop', required=True)
    # parser.add_argument('config', type=str, help='.yaml configure file path')

    parser.add_argument('matcher', type=str)
    parser.add_argument('seq_path', type=str)
    parser.add_argument('--data_path', type=str, default='data/ho3d')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda:0')

    # parser.add_argument('--resize', action='store_true')
    parser.add_argument('--resize', type=int, default=None)
    # parser.add_argument('--w_new', type=int, default=640)
    # parser.add_argument('--h_new', type=int, default=480)
    # parser.add_argument('--mask', action='store_true')
    # parser.add_argument('--depth', action='store_true')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
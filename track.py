import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torchvision

from datasets.ho3d import HO3D
from utils import project_3D_points, plot_3D_box
from baselines.pose import PoseRecover
from pose_tracking.pose_tracker import PoseTracker
from model import SparseRelPose


def main(args):
    testset = HO3D(args.data_path, args.seq_path, mode='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    device = args.device
    img_resize = args.resize
    data_path = args.data_path

    if args.model != 'relpose':
        pose_recover = PoseRecover(matcher=args.model, img_resize=img_resize, device=device)
    else:
        pose_recover = SparseRelPose(args.ckpt_path, device=device)

    K = testset[0]['intrinsic']
    pose_tracker = PoseTracker(K, args.model, pose_recover, device)

    adds, adis = [], []
    frames = []
    obj_corners = testset[0]['objCorners']

    for i, data in enumerate(tqdm(testloader)):
        image = data['color'][0]
        depth = data['depth'][0]
        mask = data['mask'][0]
        bbox = data['bbox'][0]

        pose_tracker.run(image, depth, mask, bbox)
        
        R, t = pose_tracker.get_last_pose()
        obj_corners = obj_corners @ R.mT + t

        proj_pts = project_3D_points(K, data['objCorners'][0])
        frame = plot_3D_box(proj_pts.int().numpy(), (image.permute(1, 2, 0) * 255.).numpy().astype(np.uint8), (0, 255, 0))

        proj_pts_p = project_3D_points(K, obj_corners)
        frame = plot_3D_box(proj_pts_p.int().numpy(), frame, (0, 0, 255))

        frames.append(frame)

    frames = torch.from_numpy(np.asarray(frames))
    torchvision.io.write_video(os.path.join(args.output_path, f'{args.model}_{args.seq_path}.mp4'), frames, fps=30, video_codec="libx264")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('seq_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--data_path', type=str, default='data/ho3d')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resize', type=int, default=None)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
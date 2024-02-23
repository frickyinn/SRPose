import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time

from model import PL_RelPose, keypoint_dict
# from utils.reprojection import reprojection_error
from datasets import dataset_dict
from utils.metrics import reproj, add, adi, compute_continuous_auc, relative_pose_error, rotation_angular_error
from configs.default import get_cfg_defaults


@torch.no_grad()
def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    task = config.DATASET.TASK
    dataset = config.DATASET.DATA_SOURCE
    device = args.device

    test_num_keypoints = test_num_keypoints = config.MODEL.TEST_NUM_KEYPOINTS
    
    build_fn = dataset_dict[task][dataset]
    testset = build_fn('test', config)
    # testset = Linemod(config.DATASET.DATA_ROOT, 'test', 2, config.DATASET.MIN_VISIBLE_FRACT, config.DATASET.MAX_ANGLE_ERROR)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    pl_relpose = PL_RelPose.load_from_checkpoint(args.ckpt_path)
    pl_relpose.extractor = keypoint_dict[pl_relpose.hparams['features']](max_num_keypoints=test_num_keypoints, detection_threshold=0.0).eval().to(device)
    pl_relpose.module = pl_relpose.module.eval().to(device)

    repr_errs = []
    adds, adis = [], []
    R_errs, t_errs = [], []
    R_gts, t_gts = [], []
    # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
    io_times, ex_times, com_times = [], [], []
    for i, data in enumerate(tqdm(testloader)):
        # if i >= 100:
        #     break
        image0, image1 = data['images'][0]
        K0, K1 = data['intrinsics'][0]
        T = torch.eye(4)
        T[:3, :3] = data['rotation'][0]
        T[:3, 3] = data['translation'][0]
        T = T.numpy()

        # with record_function("model_inference"):
        R_est, t_est, io_time, ex_time, com_time = pl_relpose.predict_one_data(data)
        io_times.append(io_time)
        ex_times.append(ex_time)
        com_times.append(com_time)

        t_err, R_err = relative_pose_error(T, R_est.cpu().numpy(), t_est.cpu().numpy(), ignore_gt_t_thr=0.0)

        R_errs.append(R_err)
        t_errs.append(t_err)

        R_gt = rotation_angular_error(torch.from_numpy(T[:3, :3])[None], torch.eye(3)[None])
        R_gts.append(R_gt[0])
        t_gt = torch.tensor(T[:3, 3]).norm(2)
        t_gts.append(t_gt)

        # repr_err = reprojection_error(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], K=K1, W=image1.shape[-1], H=image1.shape[-2])
        # repr_errs.append(repr_err)

        if 'point_cloud' in data:
            adds.append(add(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))
            adis.append(adi(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))
            # prjs.append(reproj(K1.numpy(), R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))

    io_times = np.array(io_times) * 1000
    ex_times = np.array(ex_times) * 1000
    com_times = np.array(com_times) * 1000

    print(f'{np.mean(io_times):.4f}')
    print(f'{np.mean(ex_times):.4f}')
    print(f'{np.mean(com_times):.4f}')
    print(f'{np.mean((io_times+ex_times+com_times)):.4f}')

    # re = np.array(repr_errs)
    # print(f'repr_err:\t{re.mean():.4f}')
    if task == 'object':
        print(f'ADD:\t\t{compute_continuous_auc(adds, np.linspace(0.0, 0.1, 1000)):.4f}')
        print(f'ADD-S\t\t{compute_continuous_auc(adis, np.linspace(0.0, 0.1, 1000)):.4f}')
        # print(f'Proj.2D:\t{compute_continuous_auc(prjs, np.linspace(0.0, 40.0, 1000)):.4f}')

    R_errs = torch.tensor(R_errs)
    t_errs = torch.tensor(t_errs)
    R_gts = torch.tensor(R_gts).rad2deg()
    t_gts = torch.tensor(t_gts)

    return R_errs, t_errs, R_gts, t_gts


def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--task', type=str, help='scene | object', required=True)
    # parser.add_argument('--dataset', type=str, help='matterport | megadepth | scannet | bop', required=True)
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('--ckpt_path', type=str, required=True)
    # parser.add_argument('--method', type=str, help='superglue | lightglue | loftr', required=True)

    # parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

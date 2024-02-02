import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from lightglue import SuperPoint
from pl_trainer import PL_LightPose
from utils.reprojection import reprojection_error
from datasets import dataset_dict
from utils.metrics import reproj, add, adi, compute_continuous_auc, relative_pose_error, rotation_angular_error
from configs.default import get_cfg_defaults


@torch.no_grad()
def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    # seed = config.RANDOM_SEED
    # seed_torch(seed)
    try:
        data_root = config.DATASET.TEST.DATA_ROOT
    except:
        data_root = config.DATASET.DATA_ROOT

    test_num_keypoints = 2048
    
    build_fn = dataset_dict[args.task][args.dataset]
    testset = build_fn('test', config)
    # testset = Linemod(config.DATASET.DATA_ROOT, 'test', 2, config.DATASET.MIN_VISIBLE_FRACT, config.DATASET.MAX_ANGLE_ERROR)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    pl_lightpose = PL_LightPose.load_from_checkpoint(args.resume)
    pl_lightpose.extractor = SuperPoint(max_num_keypoints=test_num_keypoints, detection_threshold=0.0).eval().cuda()
    pl_lightpose.module = pl_lightpose.module.eval().cuda()

    repr_errs = []
    adds, adis, prjs = [], [], []
    R_errs, t_errs = [], []
    R_gts, t_gts = [], []
    for i, data in enumerate(tqdm(testloader)):
        image0, image1 = data['images'][0]
        K0, K1 = data['intrinsics'][0]
        T = torch.eye(4)
        T[:3, :3] = data['rotation'][0]
        T[:3, 3] = data['translation'][0]
        T = T.numpy()

        R_est, t_est = pl_lightpose.predict_one_data(data)

        t_err, R_err = relative_pose_error(T, R_est.cpu().numpy(), t_est.cpu().numpy(), ignore_gt_t_thr=0.0)

        R_errs.append(R_err)
        t_errs.append(t_err)

        R_gt = rotation_angular_error(torch.from_numpy(T[:3, :3])[None], torch.eye(3)[None])
        R_gts.append(R_gt[0])
        t_gt = torch.tensor(T[:3, 3]).norm(2)
        t_gts.append(t_gt)

        repr_err = reprojection_error(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], K=K1, W=image1.shape[-1], H=image1.shape[-2])
        repr_errs.append(repr_err)

        if 'point_cloud' in data:
            adds.append(add(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))
            adis.append(adi(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))
            prjs.append(reproj(K1.numpy(), R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))

    re = np.array(repr_errs)
    print(f'repr_err:\t{re.mean():.4f}')
    if args.task == 'object':
        print(f'ADD:\t\t{compute_continuous_auc(adds, np.linspace(0.0, 0.1, 1000)):.4f}')
        print(f'ADD-S\t\t{compute_continuous_auc(adis, np.linspace(0.0, 0.1, 1000)):.4f}')
        print(f'Proj.2D:\t{compute_continuous_auc(prjs, np.linspace(0.0, 40.0, 1000)):.4f}')

    R_errs = torch.tensor(R_errs)
    t_errs = torch.tensor(t_errs)
    R_gts = torch.tensor(R_gts).rad2deg()
    t_gts = torch.tensor(t_gts)

    return R_errs, t_errs, R_gts, t_gts


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, help='scene | object', required=True)
    parser.add_argument('--dataset', type=str, help='matterport | megadepth | scannet | bop', required=True)
    parser.add_argument('--config', type=str, help='.yaml configure file path', required=True)
    parser.add_argument('--resume', type=str, required=True)
    # parser.add_argument('--method', type=str, help='superglue | lightglue | loftr', required=True)

    # parser.add_argument('--world_size', type=int, default=2)
    # parser.add_argument('--device', type=str, default='cuda:0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

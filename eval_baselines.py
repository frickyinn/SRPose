import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd

# from lightglue.utils import load_image
from configs.default import get_cfg_defaults
from datasets import dataset_dict
from baselines.pose import PoseRecover
from utils.metrics import relative_pose_error, rotation_angular_error, error_auc, add, adi, compute_continuous_auc


def main(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    task = config.DATASET.TASK
    dataset = config.DATASET.DATA_SOURCE

    # try:
    #     data_root = config.DATASET.TEST.DATA_ROOT
    # except:
    #     data_root = config.DATASET.DATA_ROOT
    
    build_fn = dataset_dict[task][dataset]
    testset = build_fn('test', config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    device = args.device
    img_resize = args.resize
    poseRec = PoseRecover(matcher=args.matcher, solver=args.solver, img_resize=img_resize, device=device)
    
    preprocess_times, extract_times, match_times, recover_times = [], [], [], []
    R_errs, t_errs = [], []
    ts_errs = []
    adds, adis = [], []
    for i, data in enumerate(tqdm(testloader)):
        if dataset == 'ho3d' and args.obj_name is not None and data['objName'][0] != args.obj_name:
            continue
        
        image0, image1 = data['images'][0].to(device)
        # if dataset == 'megadepth':
        #     image0 = load_image(os.path.join(data_root, data['pair_names'][0][0])).to(device)
        #     image1 = load_image(os.path.join(data_root, data['pair_names'][1][0])).to(device)
        # else:
        #     image0, image1 = data['images'][0].to(device)

        bbox0, bbox1 = None, None
        if task == 'object':
            bbox0, bbox1 = data['bboxes'][0]
            x1, y1, x2, y2 = bbox0
            u1, v1, u2, v2 = bbox1
            image0 = image0[:, y1:y2, x1:x2]
            image1 = image1[:, v1:v2, u1:u2]

        mask0, mask1 = None, None
        if args.mask:
            mask0, mask1 = data['masks'][0].to(device)

        depth0, depth1 = None, None
        if args.depth:
            depth0, depth1 = data['depths'][0]

        K0, K1 = data['intrinsics'][0]
        T = torch.eye(4)
        T[:3, :3] = data['rotation'][0]
        T[:3, 3] = data['translation'][0]
        T = T.numpy()

        R, t, points0, points1, preprocess_time, extract_time, match_time, recover_time = poseRec.recover(image0, image1, K0, K1, bbox0, bbox1, mask0, mask1, depth0, depth1)
        preprocess_times.append(preprocess_time)
        extract_times.append(extract_time)
        match_times.append(match_time)
        recover_times.append(recover_time)

        if np.isnan(R).any():
            R_err = 180
            R = np.identity(3)
            t_err = 180
            t = np.array([0., 0., 0.])
        else:
            t_err, R_err = relative_pose_error(T, R, t, ignore_gt_t_thr=0.0)

        R_errs.append(R_err)
        t_errs.append(t_err)

        if args.depth:
            t = np.nan_to_num(t)
            ts_errs.append(torch.tensor(T[:3, 3] - t).norm(2))

            if task == 'object':
                if np.isnan(R).any():
                    adds.append(1.)
                    adis.append(1.)
                else:
                    adds.append(add(R, t, T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))
                    adis.append(adi(R, t, T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))

    metrics = []
    values = []

    preprocess_times = np.array(preprocess_time) * 1000
    extract_times = np.array(extract_time) * 1000
    match_times = np.array(match_times) * 1000
    recover_times = np.array(recover_time) * 1000

    metrics.append('Extracting Time (ms)')
    values.append(f'{np.mean(extract_times):.1f}')
    
    metrics.append('Matching Time (ms)')
    values.append(f'{np.mean(match_times):.1f}')

    metrics.append('Recovering Time (ms)')
    values.append(f'{np.mean(recover_times):.1f}')
    
    metrics.append('Total Time (ms)')
    values.append(f'{np.mean(extract_times) + np.mean(match_times) + np.mean(recover_times):.1f}')

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([R_errs, t_errs]), axis=0)
    aucs = error_auc(pose_errors, angular_thresholds, mode='Pose estimation')  # (auc@5, auc@10, auc@20)
    for k in aucs:
        metrics.append(k)
        values.append(f'{aucs[k] * 100:.2f}')
    
    R_errs = torch.tensor(R_errs)
    t_errs = torch.tensor(t_errs)

    metrics.append('Rotation Avg. Error (°)')
    values.append(f'{R_errs.mean():.2f}')

    metrics.append('Rotation Med. Error (°)')
    values.append(f'{R_errs.median():.2f}')

    metrics.append('Rotation @30° ACC')
    values.append(f'{(R_errs < 30).float().mean() * 100:.1f}')

    metrics.append('Rotation @15° ACC')
    values.append(f'{(R_errs < 15).float().mean() * 100:.1f}')

    if args.depth:
        ts_errs = torch.tensor(ts_errs)

        metrics.append('Translation Avg. Error (m)')
        values.append(f'{ts_errs.mean():.4f}')

        metrics.append('Translation Med. Error (m)')
        values.append(f'{ts_errs.median():.4f}')
        
        metrics.append('Translation @1m ACC')
        values.append(f'{(ts_errs < 1.0).float().mean() * 100:.1f}')

        metrics.append('Translation @10cm ACC')
        values.append(f'{(ts_errs < 0.1).float().mean() * 100:.1f}')

        if task == 'object':
            metrics.append('Object ADD')
            values.append(f'{compute_continuous_auc(adds, np.linspace(0.0, 0.1, 1000)) * 100:.1f}')

            metrics.append('Object ADD-S')
            values.append(f'{compute_continuous_auc(adis, np.linspace(0.0, 0.1, 1000)) * 100:.1f}')

    res = pd.DataFrame({'Metrics': metrics, 'Values': values})
    print(res)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('matcher', type=str)
    parser.add_argument('--solver', type=str, default='procrustes')

    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--depth', action='store_true')

    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--obj_name', type=str, default=None)

    parser.add_argument('--device', type=str, default='cuda:0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

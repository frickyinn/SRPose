import argparse
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from transforms3d.quaternions import mat2quat
import pandas as pd

from model import PL_RelPose, keypoint_dict
from utils.reproject import reprojection_error, Pose, save_submission
from utils.metrics import reproj, add, adi, compute_continuous_auc, relative_pose_error, rotation_angular_error
from datasets import dataset_dict
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    pl_relpose = PL_RelPose.load_from_checkpoint(args.ckpt_path)
    pl_relpose.extractor = keypoint_dict[pl_relpose.hparams['features']](max_num_keypoints=test_num_keypoints, detection_threshold=0.0).eval().to(device)
    pl_relpose.module = pl_relpose.module.eval().to(device)

    preprocess_times, extract_times, regress_times = [], [], []
    adds, adis = [], []
    repr_errs = []
    R_errs, t_errs = [], []
    ts_errs = []
    results_dict = defaultdict(list)
    for i, data in enumerate(tqdm(testloader)):
        if dataset == 'ho3d' and args.obj_name is not None and data['objName'][0] != args.obj_name:
            continue
        image0, image1 = data['images'][0]
        K0, K1 = data['intrinsics'][0]
        T = torch.eye(4)
        T[:3, :3] = data['rotation'][0]
        T[:3, 3] = data['translation'][0]
        T = T.numpy()

        # with record_function("model_inference"):
        R_est, t_est, preprocess_time, extract_time, regress_time = pl_relpose.predict_one_data(data)
        preprocess_times.append(preprocess_time)
        extract_times.append(extract_time)
        regress_times.append(regress_time)

        t_err, R_err = relative_pose_error(T, R_est.cpu().numpy(), t_est.cpu().numpy(), ignore_gt_t_thr=0.0)

        R_errs.append(R_err)
        t_errs.append(t_err)

        ts_errs.append(torch.tensor(T[:3, 3] - t_est.cpu().numpy()).norm(2))

        if dataset == 'mapfree':
            repr_err = reprojection_error(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], K=K1, W=image1.shape[-1], H=image1.shape[-2])
            repr_errs.append(repr_err)
            R = R_est.detach().cpu().numpy()
            t = t_est.reshape(-1).detach().cpu().numpy()
            scene = data['scene_id'][0]
            estimated_pose = Pose(
                image_name=data['pair_names'][1][0],
                q=mat2quat(R).reshape(-1),
                t=t.reshape(-1),
                inliers=0
            )
            results_dict[scene].append(estimated_pose)

        if 'point_cloud' in data:
            adds.append(add(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))
            adis.append(adi(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], data['point_cloud'][0].numpy()))

    metrics = []
    values = []

    preprocess_times = np.array(preprocess_times) * 1000
    extract_times = np.array(extract_times) * 1000
    regress_times = np.array(regress_times) * 1000

    metrics.append('Extracting Time (ms)')
    values.append(f'{np.mean(extract_times):.1f}')
    
    metrics.append('Recovering Time (ms)')
    values.append(f'{np.mean(regress_times):.1f}')

    metrics.append('Total Time (ms)')
    values.append(f'{np.mean(extract_times) + np.mean(regress_times):.1f}')

    # ts_errs = np.array(ts_errs)
    # print(f'Median Trans. Error (m):\t{np.median(ts_errs):.2f}')
    # print(f'Median Rot. Error (°):\t{np.median(R_errs):.2f}')

    if task == 'object':
        metrics.append('Object ADD')
        values.append(f'{compute_continuous_auc(adds, np.linspace(0.0, 0.1, 1000)) * 100:.1f}')

        metrics.append('Object ADD-S')
        values.append(f'{compute_continuous_auc(adis, np.linspace(0.0, 0.1, 1000)) * 100:.1f}')

    if dataset == 'mapfree':
        re = np.array(repr_errs)

        metrics.append('VCRE @90px Prec.')
        values.append(f'{(re < 90).mean() * 100:.2f}')

        metrics.append('VCRE Med.')
        values.append(f'{np.median(re):.2f}')
        
        save_submission(results_dict, 'assets/new_submission.zip')

    res = pd.DataFrame({'Metrics': metrics, 'Values': values})
    print(res)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('ckpt_path', type=str)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--obj_name', type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

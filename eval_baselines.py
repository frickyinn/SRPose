import os
import argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd
from kornia.feature import LoFTR

from utils import compute_pose_errors, error_auc, rotation_angular_error
from utils.pose_solver import EssentialMatrixMetricSolver, EssentialMatrixMetricSolverMEAN, PnPSolver, ProcrustesSolver
from utils.reprojection import reprojection_error
from datasets import dataset_dict
from datasets.linemod import Linemod
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
    
    build_fn = dataset_dict[args.task][args.dataset]
    testset = build_fn('test', config)
    # testset = Linemod(config.DATASET.DATA_ROOT, 'test', 2, config.DATASET.MIN_VISIBLE_FRACT, config.DATASET.MAX_ANGLE_ERROR)
    # testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

    Solvers = {
        'ess_ransac': EssentialMatrixMetricSolver,
        'ess_mean': EssentialMatrixMetricSolverMEAN,
        'pnp': PnPSolver,
        'procrustes': ProcrustesSolver
    }
    solvers = {x: Solvers[x](config) for x in Solvers}

    R_errs = []
    t_errs = []
    R_gts = []
    t_gts = []
    repr_errs = {x: [] for x in solvers}
    for i, data in enumerate(tqdm(testset)):
        if args.dataset == 'megadepth':
            # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
            image0 = load_image(os.path.join(data_root, data['pair_names'][0])).cuda()
            image1 = load_image(os.path.join(data_root, data['pair_names'][1])).cuda()
            depth0 = torch.from_numpy(np.array(h5py.File(os.path.join(data_root, data['depth_pair_names'][0]), 'r')['depth']))
            depth1 = torch.from_numpy(np.array(h5py.File(os.path.join(data_root, data['depth_pair_names'][1]), 'r')['depth']))
            assert image0.shape[1:] == depth0.shape
            assert image1.shape[1:] == depth1.shape
        else:
            image0, image1 = data['images'].cuda()
        
        # if args.task == 'object':
        #     x1, y1, x2, y2 = data['bboxes'][0]
        #     # image0_ = image0
        #     image0 = image0[:, y1:y2, x1:x2]
        
        K0, K1 = data['intrinsics']
        T = torch.eye(4)
        T[:3, :3] = data['rotation']
        T[:3, 3] = data['translation']
        T = T.numpy()

        # extract local features
        feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
        feats1 = extractor.extract(image1)
        
        # if 'scales' in data:
        #     scales = data['scales']
        #     feats0['keypoints'] *= scales[0].unsqueeze(0).cuda()
        #     feats1['keypoints'] *= scales[1].unsqueeze(0).cuda()

        # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        # if args.task == 'object':
        #     points0[:, 0] += x1
        #     points0[:, 1] += y1

        R_err, t_err, _ = compute_pose_errors(points0.cpu().numpy(), points1.cpu().numpy(), K0, K1, T, config)
        R_errs.append(R_err)
        t_errs.append(t_err)

        R_gt = rotation_angular_error(torch.from_numpy(T[:3, :3])[None], torch.eye(3)[None])
        R_gts.append(R_gt[0])
        t_gt = torch.tensor(T[:3, 3]).norm(2)
        t_gts.append(t_gt)

        for sol in solvers:
            solver = solvers[sol]
            R_est, t_est, _ = solver.estimate_pose(points0.cpu().numpy(), points1.cpu().numpy(), {'K_color0': K0, 'K_color1': K1, 'depth0': depth0, 'depth1': depth1})
            if np.isnan(R_est).any():
                # print(i, sol, 'r')
                # if f == 1:
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(image0.permute(1, 2, 0).cpu())
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(image1.permute(1, 2, 0).cpu())
                #     plt.show()
                # if f == 2:
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(depth0)
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(depth1)
                #     plt.show()
                continue
            repr_err = reprojection_error(R_est, t_est[:, 0], T[:3, :3], T[:3, 3], K=K1, W=image1.shape[-1], H=image1.shape[-2])
            repr_errs[sol].append(repr_err)

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([R_errs, t_errs]), axis=0)
    aucs = error_auc(pose_errors, angular_thresholds, mode='lightglue')  # (auc@5, auc@10, auc@20)
    for k in aucs:
        print(f'{k}:\t{aucs[k]:.4f}')
    
    R_errs = torch.tensor(R_errs)
    t_errs = torch.tensor(t_errs)
    print(f'rotation_err_avg:\t{R_errs.mean():.2f}')
    print(f'rotation_err_med:\t{R_errs.median():.2f}')
    print(f'translation_err_avg:\t{t_errs.mean():.2f}')
    print(f'translation_err_med:\t{t_errs.median():.2f}')
    
    R_gts = torch.tensor(R_gts).rad2deg()
    t_gts = torch.tensor(t_gts)
    print(f'rel_rotation_avg:\t{R_gts.mean():.2f}')
    print(f'rel_rotation_med:\t{R_gts.median():.2f}')
    print(f'rel_translation_avg:\t{t_gts.mean():.2f}')
    print(f'rel_translation_med:\t{t_gts.median():.2f}')

    for sol in repr_errs:
        re = np.array(repr_errs[sol])
        print(f'{sol}_repr_err:\t{re.mean():.4f}')

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, help='scene | object', required=True)
    parser.add_argument('--dataset', type=str, help='matterport | megadepth | scannet | bop', required=True)
    parser.add_argument('--config', type=str, help='.yaml configure file path', required=True)
    # parser.add_argument('--resume', type=str, required=True)
    # parser.add_argument('--method', type=str, help='superglue | lightglue | loftr', required=True)

    # parser.add_argument('--world_size', type=int, default=2)
    # parser.add_argument('--device', type=str, default='cuda:0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

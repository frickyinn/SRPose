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

    test_num_keypoints = 2048
    
    build_fn = dataset_dict[args.task][args.dataset]
    testset = build_fn('test', config)
    # testset = Linemod(config.DATASET.DATA_ROOT, 'test', 2, config.DATASET.MIN_VISIBLE_FRACT, config.DATASET.MAX_ANGLE_ERROR)
    # testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    pl_lightpose = PL_LightPose.load_from_checkpoint(args.resume)
    pl_lightpose.extractor = SuperPoint(max_num_keypoints=test_num_keypoints, detection_threshold=0.0).eval().cuda()
    pl_lightpose.module = pl_lightpose.module.eval().cuda()

    repr_errs = []
    for i, data in enumerate(tqdm(testset)):
        image0, image1 = data['images']
        K0, K1 = data['intrinsics']
        T = torch.eye(4)
        T[:3, :3] = data['rotation']
        T[:3, 3] = data['translation']
        T = T.numpy()

        R_est, t_est = pl_lightpose.predict_one_data(data)

        repr_err = reprojection_error(R_est.cpu().numpy(), t_est.cpu().numpy(), T[:3, :3], T[:3, 3], K=K1, W=image1.shape[-1], H=image1.shape[-2])
        repr_errs.append(repr_err)

    re = np.array(repr_errs)
    print(f'repr_err:\t{re.mean():.4f}')

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

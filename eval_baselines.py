import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd
from kornia.feature import LoFTR

from utils import compute_pose_errors, error_auc, rotation_angular_error
from datasets import dataset_dict
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
    # testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

    R_errs = []
    t_errs = []
    R_gts = []
    t_gts = []
    for data in tqdm(testset):
        if args.dataset == 'megadepth':
            # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
            image0 = load_image(os.path.join(data_root, data['pair_names'][0])).cuda()
            image1 = load_image(os.path.join(config.DATASET.TEST.DATA_ROOT, data['pair_names'][1])).cuda()
        else:
            image0, image1 = data['images'].cuda()
        
        K0, K1 = data['intrinsics'].numpy()
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

        R_err, t_err, _ = compute_pose_errors(points0.cpu().numpy(), points1.cpu().numpy(), K0, K1, T)
        R_errs.append(R_err)
        t_errs.append(t_err)

        R_gt = rotation_angular_error(torch.from_numpy(T[:3, :3])[None], torch.eye(3)[None])
        R_gts.append(R_gt[0])
        t_gt = torch.tensor(T[:3, 3]).norm(2)
        t_gts.append(t_gt)

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([R_errs, t_errs]), axis=0)
    aucs = error_auc(pose_errors, angular_thresholds, mode='lightglue')  # (auc@5, auc@10, auc@20)
    for k in aucs:
        print(f'{k}:\t{aucs[k]:.4f}')
    
    R_gts = torch.tensor(R_gts).rad2deg()
    t_gts = torch.tensor(t_gts)
    print(f'rot_err_avg:\t{R_gts.mean():.2f}')
    print(f'rot_err_med:\t{R_gts.median():.2f}')
    print(f'trans_err_avg:\t{t_gts.mean():.2f}')
    print(f'trans_err_med:\t{t_gts.median():.2f}')

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

import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, ConcatDataset

from utils import Augmentor


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 ):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test':
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        self.augment = Augmentor(mode=='train')

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        w_new, h_new = 640, 480

        image0 = cv2.imread(img_name0)
        scale0 = torch.tensor([image0.shape[1]/w_new, image0.shape[0]/h_new], dtype=torch.float)
        image0 = cv2.resize(image0, (w_new, h_new))
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        # image0 = self.augment(image0)
        image0 = torch.from_numpy(image0).permute(2, 0, 1).float() / 255.

        image1 = cv2.imread(img_name1)
        scale1 = torch.tensor([image1.shape[1]/w_new, image1.shape[0]/h_new], dtype=torch.float)
        image1 = cv2.resize(image1, (w_new, h_new))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        # image1 = self.augment(image1)
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.

        scales = torch.stack([scale0, scale1], dim=0)
        images = torch.stack([image0, image1], dim=0)

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
        intrinsics = torch.stack([K_0, K_1], dim=0)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)

        data = {
            'images': images,
            'scales': scales,  # (2, 2): [scale_w, scale_h]
            'rotation': T_0to1[:3, :3],
            'translation': T_0to1[:3, 3],
            'intrinsics': intrinsics,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            'depth_pair_names': (self.scene_info['depth_paths'][idx0], self.scene_info['depth_paths'][idx1]),
        }

        return data
    

def build_concat_megadepth(mode, config):
    if mode == 'train':
        config = config.DATASET.TRAIN
    elif mode == 'val':
        config = config.DATASET.VAL
    elif mode == 'test':
        config = config.DATASET.TEST
    else:
        raise NotImplementedError(f'mode {mode}')

    data_root = config.DATA_ROOT
    # pose_root = config.POSE_ROOT
    npz_root = config.NPZ_ROOT
    list_path = config.LIST_PATH
    # intrinsic_path = config.INTRINSIC_PATH
    min_overlap_score = config.MIN_OVERLAP_SCORE

    with open(list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]

    datasets = []
    npz_names = [f'{n}.npz' for n in npz_names]
    for npz_name in tqdm(npz_names, desc=f'Loading MegaDepth {mode} datasets',):
        npz_path = osp.join(npz_root, npz_name)
        datasets.append(MegaDepthDataset(
            data_root,
            npz_path,
            mode=mode,
            min_overlap_score=min_overlap_score,
        ))

    return ConcatDataset(datasets)

import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset

from utils import Augmentor
# from loguru import logger

# from src.utils.dataset import read_megadepth_gray, read_megadepth_depth


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                #  img_resize=None,
                #  df=None,
                #  img_padding=False,
                #  depth_padding=False,
                #  augment_fn=None,
                #  **kwargs
                 ):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test':
            # logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        self.augment = Augmentor()

        # # parameters for image resizing, padding and depthmap padding
        # if mode == 'train':
        #     assert img_resize is not None and img_padding and depth_padding
        # self.img_resize = img_resize
        # self.df = df
        # self.img_padding = img_padding
        # self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        # self.augment_fn = augment_fn if mode == 'train' else None
        # self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        # image0, mask0, scale0 = read_megadepth_gray(
        #     img_name0, self.img_resize, self.df, self.img_padding, None)
        #     # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        # image1, mask1, scale1 = read_megadepth_gray(
        #     img_name1, self.img_resize, self.df, self.img_padding, None)
        #     # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        
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

        # # read depth. shape: (h, w)
        # if self.mode in ['train', 'val']:
        #     depth0 = read_megadepth_depth(
        #         osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
        #     depth1 = read_megadepth_depth(
        #         osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
        # else:
        #     depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
        intrinsics = torch.stack([K_0, K_1], dim=0)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        # T_1to0 = T_0to1.inverse()

        # data = {
        #     'image0': image0,  # (1, h, w)
        #     # 'depth0': depth0,  # (h, w)
        #     'image1': image1,
        #     # 'depth1': depth1,
        #     'T_0to1': T_0to1,  # (4, 4)
        #     # 'T_1to0': T_1to0,
        #     'K0': K_0,  # (3, 3)
        #     'K1': K_1,
        #     # 'scale0': scale0,  # [scale_w, scale_h]
        #     # 'scale1': scale1,
        #     'dataset_name': 'MegaDepth',
        #     'scene_id': self.scene_id,
        #     'pair_id': idx,
        #     'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
        # }

        # for LoFTR training
        # if mask0 is not None:  # img_padding is True
        #     if self.coarse_scale:
        #         [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
        #                                                scale_factor=self.coarse_scale,
        #                                                mode='nearest',
        #                                                recompute_scale_factor=False)[0].bool()
        #     data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        data = {
            # 'image0': image0,
            # 'image1': image1,
            'images': images,
            'scales': scales,  # (2, 2): [scale_w, scale_h]
            'rotation': T_0to1[:3, :3],
            'translation': T_0to1[:3, 3],
            'intrinsics': intrinsics,

            # 'scene_id': self.scene_id,
            # 'pair_id': idx,
            # 'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
        }

        return data


# def extract_keypoints(mode, output_path, extractor, dataset, device='cuda:0'):
#     for data in tqdm(dataset, desc=f'Extracting {mode} dataset keypoints'):
#         image0, image1 = data['image0'][None,], data['image1'][None,]

#         with torch.no_grad():
#             feats0 = extractor({'image': image0.to(device)})
#             feats1 = extractor({'image': image1.to(device)})

#         data['keypoints'] = torch.cat([feats0['keypoints'], feats1['keypoints']], dim=0)
#         data['descriptors'] = torch.cat([feats0['descriptors'], feats1['descriptors']], dim=0)
#         data['image_size'] = torch.stack([torch.tensor(image0.shape[-2:]), torch.tensor(image1.shape[-2:])])
#         data.pop('image0')
#         data.pop('image1')

#         scene_path = osp.join(output_path, mode, data['scene_id'].split('/')[-1])
#         os.makedirs(scene_path, exist_ok=True)
#         torch.save(data, osp.join(scene_path, f"{data['pair_id']}.pth"))


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
            # img_resize=self.mgdpt_img_resize,
            # df=self.mgdpt_df,
            # img_padding=self.mgdpt_img_pad,
            # depth_padding=self.mgdpt_depth_pad,
            # augment_fn=augment_fn,
            # coarse_scale=self.coarse_scale
        ))

    return ConcatDataset(datasets)

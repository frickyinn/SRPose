import json
from pathlib import Path
import math
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F


LINEMOD_ID_TO_NAME = {
    '000001': 'ape',
    '000002': 'benchvise',
    '000003': 'bowl',
    '000004': 'camera',
    '000005': 'can',
    '000006': 'cat',
    '000007': 'mug',
    '000008': 'driller',
    '000009': 'duck',
    '000010': 'eggbox',
    '000011': 'glue',
    '000012': 'holepuncher',
    '000013': 'iron',
    '000014': 'lamp',
    '000015': 'phone',
}


def inverse_transform(trans):
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t
    return output


class BOPDataset(Dataset):

    def __init__(self,
                 dataset_path,
                 scene_path,
                 object_id,
                 min_visible_fract,
                 rgb_postfix='.png',
                 object_scale=None
                 ):
        super().__init__()
        self.dataset_path = dataset_path
        self.scene_path = scene_path
        self.object_id = object_id

        if dataset_path.name == 'lm' or dataset_path.name == 'lmo':
            base_obj_scale = 1.0
            self.models_path = self.dataset_path / 'models'
        elif dataset_path.name == 'tless':
            base_obj_scale = 0.60
            self.models_path = self.dataset_path / 'models_reconst'
        else:
            raise ValueError(f'Unknown dataset type {dataset_path.name}')

        self.model_path = self.models_path / f'obj_{self.object_id:06d}.ply'
        self.pointcloud_path = self.dataset_path / 'models_eval' / f'obj_{self.object_id:06d}.ply'

        models_info_path = self.dataset_path / 'models_eval' / 'models_info.json'
        with open(models_info_path, 'r') as f:
            self.model_info = json.load(f)[str(object_id)]

        # self.center_object = center_object
        if object_scale is None:
            self.object_scale = base_obj_scale / self.model_info['diameter']
        else:
            self.object_scale = object_scale

        self.image_scale = 1.0
        self.bounds = torch.tensor([
            (self.model_info['min_x'], self.model_info['min_x'] + self.model_info['size_x']),
            (self.model_info['min_y'], self.model_info['min_y'] + self.model_info['size_y']),
            (self.model_info['min_z'], self.model_info['min_z'] + self.model_info['size_z']),
        ])
        self.centroid = self.bounds.mean(dim=1)

        self.depth_dir = self.scene_path / 'depth'
        self.mask_dir = self.scene_path / 'mask_visib'
        self.color_dir = self.scene_path / 'rgb'
        self.intrinsics_path = self.scene_path / 'scene_camera.json'
        self.extrinsics_path = self.scene_path / 'scene_gt.json'
        self.gt_info_path = self.scene_path / 'scene_gt_info.json'

        self.intrinsics, self.depth_scales = self.load_intrinsics(self.intrinsics_path)
        self.extrinsics, self.scene_object_inds = self.load_extrinsics(self.extrinsics_path)
        self.extrinsics = torch.stack(self.extrinsics, dim=0)
        self.gt_info = self.load_gt_info(self.gt_info_path)

        # # Compute quaternions for sampling.
        # rotation, translation = three.decompose(self.extrinsics)
        # self.quaternions = three.quaternion.mat_to_quat(rotation[:, :3, :3])

        self.depth_paths = sorted([self.depth_dir / f'{frame_ind:06d}.png'
                                   for frame_ind in self.scene_object_inds.keys()])
        self.mask_paths = [
            self.mask_dir / f'{frame_ind:06d}_{obj_ind:06d}.png'
            for frame_ind, obj_ind in self.scene_object_inds.items()
        ]
        self.color_paths = sorted([self.color_dir / f'{frame_ind:06d}{rgb_postfix}'
                                   for frame_ind in self.scene_object_inds.keys()])
        
        visib_filter = np.array(self.gt_info['visib_fract']) >= min_visible_fract
        self.color_paths = np.array(self.color_paths)[visib_filter]
        self.mask_paths = np.array(self.mask_paths)[visib_filter]
        self.depth_paths = np.array(self.depth_paths)[visib_filter]
        self.depth_scales = np.array(self.depth_scales)[visib_filter]
        for k in self.gt_info:
            self.gt_info[k] = np.array(self.gt_info[k])[visib_filter]

        self.extrinsics = np.array(self.extrinsics)[visib_filter]
        self.intrinsics = np.array(self.intrinsics)[visib_filter]

        assert len(self.depth_paths) == len(self.mask_paths)
        assert len(self.depth_paths) == len(self.color_paths)

    # def load_pointcloud(self):
    #     obj = meshutils.Object3D(self.pointcloud_path)
    #     points = torch.tensor(obj.vertices, dtype=torch.float32)
    #     points = points * self.object_scale
    #     return points

    def load_gt_info(self, path):
        with open(path, 'r') as f:
            gt_info_json = json.load(f)
            keys = sorted([int(k) for k in gt_info_json.keys()])
            gt_info = {k: [] for k in gt_info_json['0'][0]}
            for key in keys:
                value = gt_info_json[str(key)]
                obj_info = value[self.scene_object_inds[key]]
                for info_k in obj_info:
                    gt_info[info_k].append(obj_info[info_k])
        
        return gt_info

    def load_intrinsics(self, path):
        intrinsics = []
        depth_scales = []
        with open(path, 'r') as f:
            intrinsics_json = json.load(f)
            keys = sorted([int(k) for k in intrinsics_json.keys()])
            for key in keys:
                value = intrinsics_json[str(key)]
                intrinsic_3x3 = value['cam_K']
                intrinsics.append(torch.tensor(intrinsic_3x3).reshape(3, 3))
                depth_scales.append(value['depth_scale'])

        return intrinsics, depth_scales

    def load_extrinsics(self, path):
        extrinsics = []
        scene_object_inds = {}
        with open(path, 'r') as f:
            extrinsics_json = json.load(f)
            frame_inds = sorted([int(k) for k in extrinsics_json.keys()])
            for frame_ind in frame_inds:
                for obj_ind, cam_d in enumerate(extrinsics_json[str(frame_ind)]):
                    if cam_d['obj_id'] == self.object_id:
                        rotation = torch.tensor(
                            cam_d['cam_R_m2c'], dtype=torch.float32).reshape(3, 3)
                        translation = torch.tensor(cam_d['cam_t_m2c'], dtype=torch.float32) / 1000.
                        # quaternion = three.quaternion.mat_to_quat(rotation)
                        # extrinsics.append(three.to_extrinsic_matrix(translation, quaternion))
                        extrinsic = torch.eye(4)
                        extrinsic[:3, :3] = rotation
                        extrinsic[:3, 3] = translation
                        extrinsics.append(extrinsic)
                        scene_object_inds[frame_ind] = obj_ind

        return extrinsics, scene_object_inds

    def __len__(self):
        return len(self.color_paths)

    def get_ids(self):
        return [p.stem for p in self.color_paths]

    def _load_color(self, path):
        image = Image.open(path)
        new_shape = (int(image.width * self.image_scale), int(image.height * self.image_scale))
        image = image.resize(new_shape)
        image = np.array(image)
        return image

    def _load_mask(self, path):
        image = Image.open(path)
        new_shape = (int(image.width * self.image_scale), int(image.height * self.image_scale))
        image = image.resize(new_shape)
        image = np.array(image, dtype=bool)
        if len(image.shape) > 2:
            image = image[:, :, 0]
        return image

    def _load_depth(self, path):
        image = Image.open(path)
        new_shape = (int(image.width * self.image_scale), int(image.height * self.image_scale))
        image = image.resize(new_shape)
        image = np.array(image, dtype=np.float32)
        return image

    # def normalize_extrinsic(self, extrinsic):
    #     extrinsic = extrinsic.clone()
    #     if self.center_object:
    #         extrinsic = three.translate_matrix(extrinsic, -self.centroid.to(extrinsic.device))
    #     extrinsic[..., :3, 3] *= self.object_scale
    #     return extrinsic

    # def denormalize_extrinsic(self, extrinsic):
    #     extrinsic = extrinsic.clone()
    #     extrinsic[..., :3, 3] /= self.object_scale
    #     if self.center_object:
    #         extrinsic = three.translate_matrix(extrinsic, self.centroid.to(extrinsic.device))
    #     return extrinsic

    # def normalize_intrinsic(self, intrinsic):
    #     intrinsic = intrinsic.clone()
    #     intrinsic[..., :2, :] *= self.image_scale
    #     return intrinsic

    # def denormalize_intrinsic(self, intrinsic):
    #     intrinsic = intrinsic.clone()
    #     intrinsic[..., :2, :] /= self.image_scale
    #     return intrinsic

    # def sample_evenly(self, n):
    #     positions = three.extrinsic_to_position(self.extrinsics)
    #     _, inds = three.utils.farthest_points(positions,
    #                                           n_clusters=n,
    #                                           dist_func=F.pairwise_distance,
    #                                           return_center_indexes=True)
    #     return inds

    def __getitem__(self, idx):
        color = self._load_color(self.color_paths[idx])
        color = (torch.tensor(color).float() / 255.0).permute(2, 0, 1)
        mask = self._load_mask(self.mask_paths[idx])
        mask = torch.tensor(mask).bool()
        depth = self._load_depth(self.depth_paths[idx])
        depth = torch.tensor(depth) * self.object_scale * self.depth_scales[idx]

        # intrinsic = self.normalize_intrinsic(self.intrinsics[idx])
        # extrinsic = self.normalize_extrinsic(self.extrinsics[idx])

        intrinsic = torch.from_numpy(self.intrinsics[idx])
        extrinsic = torch.from_numpy(self.extrinsics[idx])

        bbox_obj = self.gt_info['bbox_obj'][idx]
        bbox_visib = self.gt_info['bbox_visib'][idx]

        bbox_obj = torch.tensor([bbox_obj[0], bbox_obj[1], bbox_obj[0]+bbox_obj[2], bbox_obj[1]+bbox_obj[3]])
        bbox_visib = torch.tensor([bbox_visib[0], bbox_visib[1], bbox_visib[0]+bbox_visib[2], bbox_visib[1]+bbox_visib[3]])

        visib_fract = self.gt_info['visib_fract'][idx]
        px_count_visib = self.gt_info['px_count_visib'][idx]

        return {
            'color': color,
            'mask': mask,
            'depth': depth,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'bbox_obj': bbox_obj,
            'bbox_visib': bbox_visib,
            'visib_fract': visib_fract,
            'px_count_visib': px_count_visib
        }


class Linemod(Dataset):
    def __init__(self, data_root, mode, object_id, min_visible_fract=0.5, max_angle_error=60):
        if mode == 'train':
            type_path = 'train_pbr'
            rgb_postfix = '.jpg'
            scene_id = object_id
        elif mode == 'val':
            type_path = 'test'
            rgb_postfix = '.png'
            scene_id = object_id
        else:
            raise NotImplementedError(f'mode {mode}')
        
        data_root = Path(data_root)
        scene_path = data_root / type_path / f'{scene_id:06d}'
        self.bop_dataset = BOPDataset(data_root, scene_path, object_id=object_id, min_visible_fract=min_visible_fract, rgb_postfix=rgb_postfix)

        angle_err = self.get_angle_error(torch.from_numpy(self.bop_dataset.extrinsics[:, :3, :3]))
        index0, index1 = torch.where(angle_err < max_angle_error)
        filter = torch.where(index0 < index1)
        self.index0, self.index1 = index0[filter], index1[filter]
        # angle_err_filtered = angle_err[row, col]

        self.indices = torch.tensor(list(zip(self.index0, self.index1)))
        if mode == 'val':
            self.indices = self.indices[torch.randperm(self.indices.size(0))[:1500]]

    def get_angle_error(self, R):
        # R: (B, 3, 3)
        residual = torch.einsum('aij,bik->abjk', R, R)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        cosine = torch.clip(cosine, -1, 1)
        R_err = torch.acos(cosine)
        angle_err = R_err * 180 / torch.pi

        return angle_err

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx0, idx1 = self.indices[idx]
        data0, data1 = self.bop_dataset[idx0], self.bop_dataset[idx1]

        images = torch.stack([data0['color'], data1['color']], dim=0)

        ex0, ex1 = data0['extrinsic'], data1['extrinsic']
        rel_ex = ex1 @ ex0.inverse()
        rel_R = rel_ex[:3, :3]
        rel_t = rel_ex[:3, 3]

        intrinsics = torch.stack([data0['intrinsic'], data1['intrinsic']], dim=0)
        bboxes = torch.stack([data0['bbox_obj'], data1['bbox_obj']])

        return {
            'images': images,
            'rotation': rel_R,
            'translation': rel_t,
            'intrinsics': intrinsics,
            'bboxes': bboxes,
        }


def build_linemod(mode, config):
    config = config.DATASET
    return Linemod(config.DATA_ROOT, mode, config.OBJECT_ID, config.MIN_VISIBLE_FRACT, config.MAX_ANGLE_ERROR)

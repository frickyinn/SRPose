from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import pickle
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, ConcatDataset

from utils.augment import Augmentor


class HO3D(Dataset):
    def __init__(self, data_root, sequence_path, mode):
        self.data_root = Path(data_root)
        self.sequence_dir = self.data_root / 'train' / sequence_path

        self.color_dir = self.sequence_dir / 'rgb'
        self.mask_dir = self.sequence_dir / 'seg'
        self.meta_dir = self.sequence_dir / 'meta'
        
        self.color_paths = list(self.color_dir.iterdir())
        self.color_paths = sorted(self.color_paths)

        self.mask_paths = [self.mask_dir / f'{x.stem}.png' for x in self.color_paths]
        self.meta_paths = [self.meta_dir / f'{x.stem}.pkl' for x in self.color_paths]

        # self.glcam_in_cvcam = torch.tensor([
        #     [1,0,0,0],
        #     [0,-1,0,0],
        #     [0,0,-1,0],
        #     [0,0,0,1]
        # ]).float()
        self.intrinsics, self.extrinsics, self.objCorners, self.objNames, valid = self._load_meta(self.meta_paths)

        self.color_paths = np.array(self.color_paths)[valid.numpy()]
        self.mask_paths = np.array(self.mask_paths)[valid.numpy()]
        self.meta_paths = np.array(self.meta_paths)[valid.numpy()]

        self.bboxes, valid = self._load_bboxes(self.mask_paths)
        self.intrinsics = self.intrinsics[valid]
        self.extrinsics = self.extrinsics[valid]
        self.objCorners = self.objCorners[valid]
        self.objNames = self.objNames[valid.numpy()]
        self.color_paths = self.color_paths[valid.numpy()]
        self.mask_paths = self.mask_paths[valid.numpy()]
        self.meta_paths = self.meta_paths[valid.numpy()]

        assert len(self.color_paths) == self.intrinsics.shape[0]
        assert len(self.objNames) == self.extrinsics.shape[0]

        self.augment = Augmentor(mode=='train')

    def __len__(self):
        return len(self.color_paths)

    def _load_bboxes(self, mask_paths):
        bboxes = []
        valid = []
        for mask_path in mask_paths:
            mask = cv2.imread(str(mask_path))
            # mask = cv2.resize(mask, (640, 480))
            w_scale, h_scale = 640 / mask.shape[1], 480 / mask.shape[0]
            obj_mask = torch.from_numpy(mask[..., 1] == 255)

            if obj_mask.float().sum() < 100:
                valid.append(False)
                continue
            valid.append(True)

            mask_inds = torch.where(obj_mask)
            x1, x2 = mask_inds[0].aminmax()
            y1, y2 = mask_inds[1].aminmax()
            bbox = torch.tensor([y1*h_scale, x1*w_scale, y2*h_scale, x2*w_scale]).int()
            bboxes.append(bbox)

        bboxes = torch.stack(bboxes)
        valid = torch.tensor(valid)

        return bboxes, valid

    def _load_meta(self, meta_paths):
        intrinsics = []
        extrinsics = []
        objCorners = []
        objNames = []
        valid = []
        for meta_path in meta_paths:
            with open(meta_path, 'rb') as f:
                anno = pickle.load(f, encoding='latin1')
            
            if anno['camMat'] is None:
                valid.append(False)
                continue
            valid.append(True)

            camMat = torch.from_numpy(anno['camMat'])
            ex = torch.eye(4)
            ex[:3, :3] = torch.from_numpy(cv2.Rodrigues(anno['objRot'])[0])
            ex[:3, 3] = torch.from_numpy(anno['objTrans'])
            # ex = self.glcam_in_cvcam @ ex
            objCorners3DRest = torch.from_numpy(anno['objCorners3DRest']).float()
            # objCorners3DRest = (ex[:3, :3] @ objCorners3DRest.T + ex[:3, 3:]).T
            objCorners3DRest = objCorners3DRest @ ex[:3, :3].T + ex[:3, 3]

            intrinsics.append(camMat)
            extrinsics.append(ex)
            objCorners.append(objCorners3DRest)
            objNames.append(anno['objName'])

        intrinsics = torch.stack(intrinsics).float()
        extrinsics = torch.stack(extrinsics).float()
        objCorners = torch.stack(objCorners)
        objNames = np.array(objNames)
        valid = torch.tensor(valid)

        return intrinsics, extrinsics, objCorners, objNames, valid

    def __getitem__(self, idx):
        color = cv2.imread(str(self.color_paths[idx]))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        # color = self.augment(color)
        color = (torch.tensor(color).float() / 255.0).permute(2, 0, 1)

        bbox = self.bboxes[idx]

        intrinsic = self.intrinsics[idx]
        extrinsic = self.extrinsics[idx]
        objCorners = self.objCorners[idx]
        objName = self.objNames[idx]

        return {
            'color': color,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'objCorners': objCorners,
            'bbox': bbox,
            'color_path': str(self.color_paths[idx]).split('/', 2)[-1],
            'objName': objName,
        }


class HO3DPair(Dataset):
    def __init__(self, data_root, mode, sequence_id, max_angle_error):
        self.ho3d_dataset = HO3D(data_root, sequence_id, mode)

        angle_err = self.get_angle_error(self.ho3d_dataset.extrinsics[:, :3, :3])
        index0, index1 = torch.where(angle_err < max_angle_error)
        filter = torch.where(index0 < index1)
        self.index0, self.index1 = index0[filter], index1[filter]
        # angle_err_filtered = angle_err[row, col]

        self.indices = torch.tensor(list(zip(self.index0, self.index1)))
        if mode == 'val' or mode == 'test':
            self.indices = self.indices[torch.randperm(self.indices.size(0))[:1500]]

    def get_angle_error(self, R):
        # R: (B, 3, 3)
        residual = torch.einsum('aij,bik->abjk', R, R)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        cosine = torch.clip(cosine, -1, 1)
        R_err = torch.acos(cosine)
        angle_err = R_err.rad2deg()

        return angle_err
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx0, idx1 = self.indices[idx]
        data0, data1 = self.ho3d_dataset[idx0], self.ho3d_dataset[idx1]

        images = torch.stack([data0['color'], data1['color']], dim=0)

        ex0, ex1 = data0['extrinsic'], data1['extrinsic']
        rel_ex = ex1 @ ex0.inverse()
        rel_R = rel_ex[:3, :3]
        rel_t = rel_ex[:3, 3]

        intrinsics = torch.stack([data0['intrinsic'], data1['intrinsic']], dim=0)
        bboxes = torch.stack([data0['bbox'], data1['bbox']])
        objCorners = torch.stack([data0['objCorners'], data1['objCorners']])

        return {
            'images': images,
            'rotation': rel_R,
            'translation': rel_t,
            'intrinsics': intrinsics,
            'bboxes': bboxes,
            'objCorners': objCorners,
            'pair_names': (data0['color_path'], data1['color_path']),
            'objName': data0['objName']
        }
    

class HO3DfromJson(Dataset):
    def __init__(self, data_root, json_path):
        self.data_root = Path(data_root)
        with open(json_path, 'r') as f:
            self.scene_info = json.load(f)

        self.obj_names = [
            '003_cracker_box',
            '006_mustard_bottle',
            '011_banana',
            '025_mug',
            '037_scissors'
        ]
        self.object_points = {obj: np.loadtxt(self.data_root / 'models' / obj / 'points.xyz')  for obj in self.obj_names}

    def _load_color(self, path):
        color = cv2.imread(path)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color
    
    def _load_mask(self, path):
        mask_path = str(path).replace('rgb', 'seg').replace('.jpg', '.png')
        mask = cv2.imread(str(mask_path))
        mask = cv2.resize(mask, (640, 480))
        mask = mask[..., 1] == 255
        return mask
    
    def _load_depth(self, path):
        depth_scale = 0.00012498664727900177

        depth_path = str(path).replace('rgb', 'depth').replace('.jpg', '.png')
        depth_img = cv2.imread(depth_path)

        dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
        dpt = dpt * depth_scale

        return dpt

    def __len__(self):
        return len(self.scene_info)

    def __getitem__(self, idx):
        info = self.scene_info[str(idx)]
        pair_names = info['pair_names']

        image0 = self._load_color(str(self.data_root / pair_names[0]))
        image0 = (torch.tensor(image0).float() / 255.0).permute(2, 0, 1)
        image1 = self._load_color(str(self.data_root / pair_names[1]))
        image1 = (torch.tensor(image1).float() / 255.0).permute(2, 0, 1)
        images = torch.stack([image0, image1], dim=0)
        
        mask0 = self._load_mask(str(self.data_root / pair_names[0]))
        mask0 = torch.from_numpy(mask0)
        mask1 = self._load_mask(str(self.data_root / pair_names[1]))
        mask1 = torch.from_numpy(mask1)
        masks = torch.stack([mask0, mask1], dim=0)

        depth0 = self._load_depth(str(self.data_root / pair_names[0]))
        depth0 = torch.from_numpy(depth0)
        depth1 = self._load_depth(str(self.data_root / pair_names[1]))
        depth1 = torch.from_numpy(depth1)
        depths = torch.stack([depth0, depth1], dim=0)

        rotation = torch.tensor(info['rotation']).reshape(3, 3)
        translation = torch.tensor(info['translation'])
        intrinsics = torch.tensor(info['intrinsics']).reshape(2, 3, 3)
        bboxes = torch.tensor(info['bboxes'])
        objCorners = torch.tensor(info['objCorners'])

        return {
            'images': images,
            'masks': masks,
            'depths': depths,
            'rotation': rotation,
            'translation': translation,
            'intrinsics': intrinsics,
            'bboxes': bboxes,
            'objCorners': objCorners,
            'objName': info['objName'][0],
            'point_cloud': self.object_points[info['objName'][0]]
        }
    

def build_ho3d(mode, config):
    config = config.DATASET

    data_root = config.DATA_ROOT
    seq_id_list = [x.stem for x in (Path(data_root) / 'train').iterdir()]
    val_id_list = ['BB14', 'SMu1', 'MC1', 'GSF14', 'SM2', 'SM3', 'SM4', 'SM5', 'MC2', 'MC4', 'MC5', 'MC6']
    for val_id in val_id_list:
        seq_id_list.remove(val_id)

    if mode == 'train':
        datasets = []
        for seq_id in tqdm(seq_id_list, desc=f'Loading HO3D {mode} dataset'):
            datasets.append(HO3DPair(data_root, mode, seq_id, config.MAX_ANGLE_ERROR))
        return ConcatDataset(datasets)
    
    elif mode == 'test' or mode == 'val':
        # datasets = []
        # for seq_id in tqdm(val_id_list[:5], desc=f'Loading HO3D {mode} dataset'):
        #     datasets.append(HO3DPair(data_root, mode, seq_id, config.MAX_ANGLE_ERROR))
        # return ConcatDataset(datasets)
    
        return HO3DfromJson(config.DATA_ROOT, config.JSON_PATH)
    
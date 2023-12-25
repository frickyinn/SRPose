import numpy as np
import cv2
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset

from utils import rotation_matrix_from_quaternion, Augmentor


class Matterport3D(Dataset):
    def __init__(self, data_root, mode='train'):
        data_root = Path(data_root)
        json_path = data_root / 'mp3d_planercnn_json' / f'cached_set_{mode}.json'

        scene_info = {'images': [], 'rotation': [], 'translation': [], 'intrinsics': []}

        with open(json_path) as f:
            split = json.load(f)
        
        for _, data in enumerate(split['data']):
            images = []
            for imgnum in ['0', '1']:
                img_name = data_root / '/'.join(data[imgnum]['file_name'].split('/')[6:])
                images.append(img_name)
            
            rel_rotation = data['rel_pose']['rotation']
            rel_translation = data['rel_pose']['position']
            intrinsic = [
                [517.97, 0, 320],
                [0, 517.97, 240],
                [0, 0, 1]
            ]
            intrinsics = [intrinsic, intrinsic]

            scene_info['images'].append(images)
            scene_info['rotation'].append(rel_rotation)
            scene_info['translation'].append(rel_translation)
            scene_info['intrinsics'].append(intrinsics)

        scene_info['rotation'] = torch.tensor(scene_info['rotation'])
        scene_info['translation'] = torch.tensor(scene_info['translation'])
        scene_info['intrinsics'] = torch.tensor(scene_info['intrinsics'])

        self.scene_info = scene_info
        self.augment = Augmentor()

        self.is_training = mode == 'train'

    def __len__(self):
        return len(self.scene_info['images'])

    def __getitem__(self, idx):
        img_names = self.scene_info['images'][idx]
        rotation = self.scene_info['rotation'][idx]
        translation = self.scene_info['translation'][idx]
        intrinsics = self.scene_info['intrinsics'][idx]

        images = []
        for i in range(2):
            image = cv2.imread(str(img_names[i]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.augment(image)
            image = torch.from_numpy(image).permute(2, 0, 1)
            images.append(image)
        images = torch.stack(images)
        images = images.float() / 255.

        rotation = -rotation if rotation[0] < 0 else rotation
        rotation /= rotation.norm(2)
        rotation = rotation_matrix_from_quaternion(rotation[None,])[0]

        # if self.is_training and np.random.rand() > 0.5:
        #     images = images[[1, 0]]

        #     rotation = rotation.mT
        #     translation = -rotation @ translation.unsqueeze(-1)
        #     translation = translation[:, 0]

        return {
            'images': images,
            'rotation': rotation,
            'translation': translation,
            'intrinsics': intrinsics,
        }


def build_matterport(mode, config):
    return Matterport3D(config.DATASET.DATA_ROOT, mode)

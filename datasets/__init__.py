from .bop import BOPPair
from .matterport3d import Matterport3D
from .augmentation import RGBDAugmentor

dataset_dict = {
    'scene': {
        'matterport': Matterport3D,
    },
    'object': {
        'bop': BOPPair,
    }
}

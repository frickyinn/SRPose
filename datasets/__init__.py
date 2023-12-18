from .bop import BOPPair
from .matterport3d import Matterport3D
from .augmentation import RGBDAugmentor

dataset_dict = {
    'matterport': Matterport3D,
    'bop': BOPPair,
}

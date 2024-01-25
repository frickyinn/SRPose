from .matterport import build_matterport
from .linemod import build_linemod
from .megadepth import build_concat_megadepth
from .scannet import build_concat_scannet
from .ho3d import build_ho3d
from .sampler import RandomConcatSampler

dataset_dict = {
    'scene': {
        'matterport': build_matterport,
        'megadepth': build_concat_megadepth,
        'scannet': build_concat_scannet,
    },
    'object': {
        'linemod': build_linemod,
        'ho3d': build_ho3d,
    }
}

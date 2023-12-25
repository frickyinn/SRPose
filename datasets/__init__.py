from .matterport import build_matterport
from .linemod import build_linemod
from .megadepth import build_concat_megadepth
from .scannet import build_concat_scannet
from .sampler import RandomConcatSampler

dataset_dict = {
    'scene': {
        'matterport': build_matterport,
        'megadepth': build_concat_megadepth,
        'scannet': build_concat_scannet,
    },
    'object': {
        'linemod': build_linemod,
    }
}

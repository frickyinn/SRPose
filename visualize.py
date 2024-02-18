import argparse
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightglue import SuperPoint

from datasets import dataset_dict
from pl_trainer import PL_LightPose
from configs.default import get_cfg_defaults


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features


def get_model(args):
    config = get_cfg_defaults()
    config.merge_from_file(args.config)

    task = config.DATASET.TASK
    dataset = config.DATASET.DATA_SOURCE
    test_num_keypoints = config.MODEL.TEST_NUM_KEYPOINTS

    build_fn = dataset_dict[task][dataset]
    testset = build_fn('test', config)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    pl_lightpose = PL_LightPose.load_from_checkpoint(args.ckpt_path)
    pl_lightpose.extractor = SuperPoint(max_num_keypoints=test_num_keypoints, detection_threshold=0.0).eval()

    return pl_lightpose, testloader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='.yaml configure file path')
    parser.add_argument('--ckpt_path', type=str, required=True)

    return parser

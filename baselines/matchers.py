import torch
import time

from lightglue import LightGlue as LightGlue_
from lightglue import SuperPoint
from lightglue.utils import rbd
from kornia.feature import LoFTR as LoFTR_


def image_rgb2gray(image):
    # in: torch.tensor - (3, H, W)
    # out: (1, H, W)
    image = image[0] * 0.3 + image[1] * 0.59 + image[2] * 0.11
    return image[None]


class LightGlue():
    def __init__(self, num_keypoints=2048, device='cuda'):
        self.extractor = SuperPoint(max_num_keypoints=num_keypoints).eval().to(device)  # load the extractor
        self.matcher = LightGlue_(features='superpoint').eval().to(device)  # load the matcher
        self.device = device

    @torch.no_grad()
    def match(self, image0, image1):
        start_time = time.time()

        # image: torch.tensor - (3, H, W)
        image0 = image0.to(self.device)
        image1 = image1.to(self.device)

        preprocess_time = time.time()

        # extract local features
        feats0 = self.extractor.extract(image0)  # auto-resize the image, disable with resize=None
        feats1 = self.extractor.extract(image1)

        extract_time = time.time()

        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        match_time = time.time()

        return points0, points1, preprocess_time-start_time, extract_time-preprocess_time, match_time-extract_time


class LoFTR():
    def __init__(self, pretrained='indoor', device='cuda'):
        self.loftr = LoFTR_(pretrained=pretrained).eval().to(device)
        self.device = device

    @torch.no_grad()
    def match(self, image0, image1):
        start_time = time.time()

        # image: torch.tensor - (3, H, W)
        image0 = image_rgb2gray(image0)[None].to(self.device)
        image1 = image_rgb2gray(image1)[None].to(self.device)

        preprocess_time = time.time()

        extract_time = time.time()

        out = self.loftr({'image0': image0, 'image1': image1})
        points0, points1 = out['keypoints0'], out['keypoints1']
        
        match_time = time.time()
        return points0, points1, preprocess_time-start_time, extract_time-preprocess_time, match_time-extract_time

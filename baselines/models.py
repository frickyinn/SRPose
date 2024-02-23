import yaml
import numpy as np
import cv2
import torch
from PIL import Image
import time
import pdb

from lightglue import LightGlue as LightGlue_
from lightglue import SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
from kornia.feature import LoFTR as LoFTR_

from .repo.SuperGluePretrainedNetwork.models.matching import Matching
from .repo.SGMNet.components import load_component
from .repo.MatchFormer.model.matchformer import Matchformer as MatchFormer_
from .repo.aspanformer.src.ASpanFormer.aspanformer import ASpanFormer as ASpanFormer_
from .repo.aspanformer.src.config.default import get_cfg_defaults
from .repo.aspanformer.src.utils.misc import lower_config
from .repo.DKM.dkm import DKMv3_indoor, DKMv3_outdoor

# from .repo.essnet.networks import EssNet as EssNet_
from .repo.essnet.utils.common.relapose_config import RPEvalConfig
from .repo.essnet.utils.eval.localize import decompose_essential_matrix
# from .repo.QuadTreeAttention.FeatureMatching.src.loftr import LoFTR as QuadLoFTR
# from .repo.pats.models.pats import PATS as PATS_
# from .repo.POPE.src.matcher import Matcher, default_cfg



# class EssNet():
#     def __init__(self, device='cuda'):
#         ckpt_dir = 'RelPoseRepo/weights/essnet/essnet_scan95ep.pth'
#         config = RPEvalConfig(data_root=None,
#                             # datasets=['CambridgeLandmarks'],
#                             # incl_sces=['ShopFacade'],
#                             # pair_txt=test_pair_txt,
#                             rescale=224,
#                             crop=224,
#                             network='EssNet',
#                             resume=ckpt_dir,
#                             # odir='../output/regression_models/example'
#                             )

#         self.regressor = EssNet_(config).eval().to(device)
#         self.device = device

#     @torch.no_grad()
#     def regress(self, image0, image1):
#         image0 = image0.to(self.device)[None]
#         image1 = image1.to(self.device)[None]

#         E = self.regressor(image0, image1)
#         E = E.cpu().data.numpy().reshape((3,3))
#         (t, R0, R1) = decompose_essential_matrix(E)

#         return R0, R1, t


def image_rgb2gray(image):
    # in: torch.tensor - (3, H, W)
    # out: (1, H, W)
    image = image[0] * 0.3 + image[1] * 0.59 + image[2] * 0.11
    return image[None]


class SuperGlue():
    def __init__(self, nms_radius=4, keypoint_threshold=0.005, max_keypoints=2048, superglue='indoor', sinkhorn_iterations=20, match_threshold=0.2, device='cuda'):
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self.matching = Matching(config).eval().to(device)
        self.device = device

    @torch.no_grad()
    def match(self, image0, image1):
        st_time = time.time()
        
        # image: torch.tensor - (3, H, W)
        image0 = image_rgb2gray(image0)[None].to(self.device)
        image1 = image_rgb2gray(image1)[None].to(self.device)

        io_time = time.time()

        ex_time = time.time()

        pred = self.matching({'image0': image0, 'image1': image1})
        pred = {k: v[0] for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        com_time = time.time()

        return mkpts0, mkpts1, io_time-st_time, ex_time-io_time, com_time-ex_time


# class POPE():
#     def __init__(self, device):
#         self.matcher = Matcher(config=default_cfg)
#         # we set strict to False
#         self.matcher.load_state_dict(torch.load("weights/matcher.pth")['state_dict'], strict=False)
#         self.matcher = self.matcher.eval().to(device)
#         self.device = device

#     @torch.no_grad()
#     def match(self, image0, image1):
#         # image: torch.tensor - (3, H, W)
#         image0 = image_rgb2gray(image0)[None].to(self.device)
#         image1 = image_rgb2gray(image1)[None].to(self.device)

#         batch = {'image0': image0, 'image1': image1}
#         matcher(batch)    
#         mkpts0 = batch['mkpts0_f']
#         mkpts1 = batch['mkpts1_f']
#         confidences = batch["mconf"]
        
#         import pdb
#         pdb.set_trace()

#         return mkpts0, mkpts1

# class SGMNet():
#     def __init__(self, config_path):
#         with open(config_path, 'r') as f:
#             self.config = yaml.load(f)
#         self.extractor=load_component('extractor',self.config['extractor']['name'],self.config['extractor'])
#         self.matcher=load_component('matcher',self.config['matcher']['name'],self.config['matcher'])

#     @torch.no_grad()
#     def match(self, img1_path, img2_path):
#         img1,img2=cv2.imread(img1_path),cv2.imread(img2_path)
#         size1,size2=np.flip(np.asarray(img1.shape[:2])),np.flip(np.asarray(img2.shape[:2]))
#         kpt1,desc1=self.extractor.run(img1_path)
#         kpt2,desc2=self.extractor.run(img2_path)
        
#         matcher=load_component('matcher',self.config['matcher']['name'],self.config['matcher'])
#         test_data={'x1':kpt1,'x2':kpt2,'desc1':desc1,'desc2':desc2,'size1':size1,'size2':size2}
#         corr1,corr2= matcher.run(test_data)

#         return corr1, corr2
    

class LightGlue():
    def __init__(self, num_keypoints=2048, device='cuda'):
        self.extractor = SuperPoint(max_num_keypoints=num_keypoints).eval().to(device)  # load the extractor
        self.matcher = LightGlue_(features='superpoint').eval().to(device)  # load the matcher
        self.device = device

    @torch.no_grad()
    def match(self, image0, image1):
        st_time = time.time()

        # image: torch.tensor - (3, H, W)
        image0 = image0.to(self.device)
        image1 = image1.to(self.device)

        io_time = time.time()

        # extract local features
        feats0 = self.extractor.extract(image0, resize=None)  # auto-resize the image, disable with resize=None
        feats1 = self.extractor.extract(image1, resize=None)

        ex_time = time.time()

        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        com_time = time.time()

        return points0, points1, io_time-st_time, ex_time-io_time, com_time-ex_time


class SGMNet():
    def __init__(self, config_path='RelPoseRepo/repo/SGMNet/demo/configs/sgm_config.yaml', device='cuda'):
        with open(config_path, 'r') as f:
            demo_config = yaml.safe_load(f)

        self.extractor=load_component('extractor',demo_config['extractor']['name'],demo_config['extractor'])
        self.matcher = load_component('matcher', demo_config['matcher']['name'], demo_config['matcher'])
        self.device = device

    @torch.no_grad()
    def match(self, image0, image1):
        st_time = time.time()

        # image: torch.tensor - (3, H, W)
        size1,size2=np.flip(np.asarray([image0.shape[-2],image0.shape[-1]])), np.flip(np.asarray([image1.shape[-2],image1.shape[-1]]))

        image0 = image_rgb2gray(image0)
        image1 = image_rgb2gray(image1)

        image0 = (image0[0] * 255.).int().cpu().numpy().astype(np.uint8)
        image1 = (image1[0] * 255.).int().cpu().numpy().astype(np.uint8)

        io_time = time.time()

        kpt1,desc1=self.extractor.run(image0)
        kpt2,desc2=self.extractor.run(image1)
        # pdb.set_trace()

        ex_time = time.time()

        test_data={'x1':kpt1,'x2':kpt2,'desc1':desc1,'desc2':desc2,'size1':size1,'size2':size2}
        corr1,corr2= self.matcher.run(test_data)
        
        com_time = time.time()
        return torch.from_numpy(corr1), torch.from_numpy(corr2), io_time-st_time, ex_time-io_time, com_time-ex_time
    

class LoFTR():
    def __init__(self, pretrained='indoor_new', device='cuda'):
        self.loftr = LoFTR_(pretrained=pretrained).eval().to(device)
        self.device = device

    @torch.no_grad()
    def match(self, image0, image1):
        st_time = time.time()

        # image: torch.tensor - (3, H, W)
        image0 = image_rgb2gray(image0)[None].to(self.device)
        image1 = image_rgb2gray(image1)[None].to(self.device)

        io_time = time.time()

        ex_time = time.time()

        out = self.loftr({'image0': image0, 'image1': image1})
        points0, points1 = out['keypoints0'], out['keypoints1']
        
        com_time = time.time()
        return points0, points1, io_time-st_time, ex_time-io_time, com_time-ex_time
    

class MatchFormer():
    def __init__(self, config, device='cuda'):
        self.matchformer = MatchFormer_(config).eval().to(self.device)
        self.device = device
    
    @torch.no_grad()
    def match(self, image0, image1):
        image0, image1 = image0[None].to(self.device), image1[None].to(self.device)
        out = self.matchformer({'image0': image0, 'image1': image1})
        points0, points1 = out['mkpts0_f'], out['mkpts1_f']

        return points0, points1


class ASpanFormer():
    def __init__(self, config_path='RelPoseRepo/repo/aspanformer/configs/aspan/indoor/aspan_test.py',
                 weights_path='RelPoseRepo/weights/aspanformer/indoor.ckpt', device='cuda'):
        
        config = get_cfg_defaults()
        
        from .repo.aspanformer.src.config.default import _CN as cfg
        cfg.ASPAN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

        cfg.ASPAN.MATCH_COARSE.BORDER_RM = 0
        cfg.ASPAN.COARSE.COARSEST_LEVEL= [15,20]
        cfg.ASPAN.COARSE.TRAIN_RES = [480,640]
        
        _config = lower_config(cfg)

        self.aspanformer = ASpanFormer_(config=_config['aspan'])
        state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
        self.aspanformer.load_state_dict(state_dict,strict=False)
        self.aspanformer = self.aspanformer.eval().to(device)
        self.device = device

    @torch.no_grad()
    def match(self, image0, image1):
        st_time = time.time()

        # image: torch.tensor - (3, H, W)
        image0 = image_rgb2gray(image0)[None].to(self.device)
        image1 = image_rgb2gray(image1)[None].to(self.device)

        io_time = time.time()

        ex_time = time.time()

        data = {'image0': image0, 'image1': image1}
        self.aspanformer(data)
        corr0, corr1 = data['mkpts0_f'], data['mkpts1_f']

        com_time = time.time()

        return corr0, corr1, io_time-st_time, ex_time-io_time, com_time-ex_time


# class QuadTree():
#     def __init__(self, config, weight_path):
#         matcher = QuadLoFTR(config)
#         state_dict = torch.load(weight_path, map_location="cpu")["state_dict"]
#         matcher.load_state_dict(state_dict, strict=True)
#         self.matcher = matcher.eval().cuda()

#     @torch.no_grad()
#     def match(self, image0, image1):
#         # image: torch.tensor - (3, H, W)
#         image0 = image_rgb2gray(image0)[None].to(self.device)
#         image1 = image_rgb2gray(image1)[None].to(self.device)

#         batch = {
#             "image0": image0,
#             "image1": image1,
#         }
#         self.matcher(batch)

#         query_kpts = batch["mkpts0_f"].cpu().numpy()
#         ref_kpts = batch["mkpts1_f"].cpu().numpy()

#         return query_kpts, ref_kpts
    

class DKM():
    def __init__(self, pretrained='indoor', device='cuda'):
        if pretrained == 'indoor':
            self.dkm = DKMv3_indoor(device=device).eval()
        elif pretrained == 'outdoor':
            self.dkm = DKMv3_outdoor(device=device).eval()

        self.device = device

    @torch.no_grad()
    def match(self, image0, image1):
        st_time = time.time()

        image0 = tensor2Image(image0)
        image1 = tensor2Image(image1)

        W_A, H_A = image0.size
        W_B, H_B = image1.size

        io_time = time.time()

        ex_time = time.time()

        # Match
        warp, certainty = self.dkm.match(image0, image1, device=self.device)
        # Sample matches for estimation
        matches, certainty = self.dkm.sample(warp, certainty)
        kpts1, kpts2 = self.dkm.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

        com_time = time.time()

        return kpts1, kpts2, io_time-st_time, ex_time-io_time, com_time-ex_time


def tensor2Image(image):
    image = (image.permute(1, 2, 0) * 255).int()
    image = image.cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    return image


# class PATS():
#     def __init__(self, param, device='cuda'):
#         pats = PATS_(param)
#         pats.load_state_dict()
#         self.pats = pats.eval().to(device)

#     @torch.no_grad()
#     def match(self, image0, image1):
#         image0, image1 = image0[None].to(self.device), image1[None].to(self.device)

#         result = self.pats({'image0': image0, 'image1': image1})

#         kp0 = result["matches_l"].cpu().numpy()
#         kp1 = result["matches_r"].cpu().numpy()

#         return kp0, kp1



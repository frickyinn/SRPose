import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import rotation_matrix_from_ortho6d

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('medium')


# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
# def normalize_keypoints(
#     kpts: torch.Tensor, size: Optional[torch.Tensor] = None
# ) -> torch.Tensor:
#     if size is None:
#         size = 1 + kpts.max(-2).values - kpts.min(-2).values
#     elif not isinstance(size, torch.Tensor):
#         size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
#     size = size.to(kpts)
#     shift = size / 2
#     scale = size.max(-1).values / 2
#     kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
#     return kpts

# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(kpts, intrinsics):
    # kpts: (B, M, 2)
    # intrinsics: (B, 3, 3)

    b, m, _ = kpts.shape
    kpts = torch.cat([kpts, torch.ones((b, m, 1), device=kpts.device)], dim=2)
    kpts = intrinsics.inverse() @ kpts.mT
    kpts = kpts.mT
    kpts = kpts[..., :2]

    return kpts


# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def cosine_similarity(x, y):
    sim = torch.einsum('...id,...jd->...ij', x / x.norm(2, -1, keepdim=True), y / y.norm(2, -1, keepdim=True))
    sim = (sim + 1) / 2
    return sim


def pad_to_length(x: torch.Tensor, length: int) -> Tuple[torch.Tensor]:
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(
        *x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
    )
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask
        

def gather(x: torch.Tensor, indices: torch.tensor):
    b, _, n = x.shape
    bs = torch.arange(b).reshape(b, 1, 1)
    ns = torch.arange(n)
    return x[bs, indices.unsqueeze(-1), ns]


class Attention(nn.Module):
    def __init__(self, allow_flash: bool = True) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if self.has_sdp:
                args = [x.contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
            else:
                assert mask is None
                q, k, v = [x.transpose(-2, -3).contiguous() for x in [q, k, v]]
                m = self.flash_(q, torch.stack([k, v], 2))
                return m.transpose(-2, -3).to(q.dtype).clone()
        elif self.has_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q += encoding
        k += encoding

        # s = q.shape[-1] ** -0.5
        # sim = torch.einsum("...id,...jd->...ij", q, k) * s
        # if mask is not None:
        #     sim.masked_fill(~mask.unsqueeze(1), -float("inf"))
        # attn = F.softmax(sim, -1)
        # context = torch.einsum("...ij,...jd->...id", attn, v)
        context = self.inner_attn(q, k, v, mask=mask)

        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, match: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:

        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        
        qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
        sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
        if mask is not None:
            sim = sim.masked_fill(~mask.unsqueeze(1), -float("inf"))
        
        assert len(match.shape) == 3
        match = match.unsqueeze(1)
        sim = sim * match
        
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
        m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
        if mask is not None:
            m0, m1 = m0.nan_to_num(), m1.nan_to_num()

        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        match,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, match, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1, match)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, match, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, match, mask)


class LightPose(nn.Module):
    default_conf = {
        "name": "lightpose",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 3,
        "num_heads": 4,
        "pct_pruning": 0,
        "task": "scene",
        "mp": False,  # enable mixed precision
        "weights": None,
    }

    # # Point pruning involves an overhead (gather).
    # # Therefore, we only activate it if there are enough keypoints.
    # pruning_keypoint_thresholds = {
    #     "cpu": -1,
    #     "mps": -1,
    #     "cuda": 1024,
    #     "flash": 1536,
    # }

    required_data_keys = ["image0", "image1"]

    features = {
        "superpoint": {
            "weights": "superpoint_lightglue",
            "input_dim": 256,
        },
        "disk": {
            "weights": "disk_lightglue",
            "input_dim": 128,
        },
        "aliked": {
            "weights": "aliked_lightglue",
            "input_dim": 128,
        },
        "sift": {
            "weights": "sift_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
    }

    def __init__(self, features="superpoint", **conf) -> None:
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        if features is not None:
            if features not in self.features:
                raise ValueError(
                    f"Unsupported features: {features} not in "
                    f"{{{','.join(self.features)}}}"
                )
            for k, v in self.features[features].items():
                setattr(conf, k, v)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = nn.Linear(
            2 + 2 * self.conf.add_scale_ori, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h) for _ in range(n)]
        )

        self.rotation_regressor = nn.Sequential(
            nn.Linear(conf.input_dim*2, conf.input_dim), 
            nn.ReLU(), 
            nn.Linear(conf.input_dim, conf.input_dim//2), 
            nn.ReLU(), 
            nn.Linear(conf.input_dim//2, 6),
        )

        self.translation_regressor = nn.Sequential(
            nn.Linear(conf.input_dim*2, conf.input_dim), 
            nn.ReLU(), 
            nn.Linear(conf.input_dim, conf.input_dim//2), 
            nn.ReLU(), 
            nn.Linear(conf.input_dim//2, 3),
        )

        # self.regressor = nn.Sequential(
        #     nn.Linear(conf.input_dim*2, conf.input_dim), 
        #     nn.ReLU(), 
        #     nn.Linear(conf.input_dim, conf.input_dim//2), 
        #     nn.ReLU(), 
        #     nn.Linear(conf.input_dim//2, 9),
        # )

        # state_dict = None
        # if features is not None:
        #     fname = f"{conf.weights}_{self.version.replace('.', '-')}.pth"
        #     state_dict = torch.hub.load_state_dict_from_url(
        #         self.url.format(self.version, features), file_name=fname
        #     )
        #     self.load_state_dict(state_dict, strict=False)
        # elif conf.weights is not None:
        #     path = Path(__file__).parent
        #     path = path / "weights/{}.pth".format(self.conf.weights)
        #     state_dict = torch.load(str(path), map_location="cpu")

        # if state_dict:
        #     # rename old state dict entries
        #     for i in range(self.conf.n_layers):
        #         pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
        #         state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        #         pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
        #         state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        #     self.load_state_dict(state_dict, strict=False)

        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None

    def compile(
        self, mode="reduce-overhead", static_lengths=[256, 512, 768, 1024, 1280, 1536]
    ):
        for i in range(self.conf.n_layers):
            self.transformers[i].masked_forward = torch.compile(
                self.transformers[i].masked_forward, mode=mode, fullgraph=True
            )

        self.static_lengths = static_lengths

    def forward(self, data: dict) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
                image: [B x C x H x W] or image_size: [B x 2]
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
                image: [B x C x H x W] or image_size: [B x 2]
        Output 
        """
        with torch.autocast(enabled=self.conf.mp, device_type="cuda"):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        intrinsic0, intrinsic1 = data0["intrinsics"], data1["intrinsics"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        # device = kpts0.device
        # size0, size1 = data0.get("image_size"), data1.get("image_size")
        # kpts0 = normalize_keypoints(kpts0, size0).clone()
        # kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            kpts0 = torch.cat(
                [kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
            kpts1 = torch.cat(
                [kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
        desc0 = data0["descriptors"].detach().contiguous()
        desc1 = data1["descriptors"].detach().contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim

        # if torch.is_autocast_enabled():
        #     desc0 = desc0.half()
        #     desc1 = desc1.half()

        mask0, mask1 = None, None
        c = max(m, n)
        do_compile = self.static_lengths and c <= max(self.static_lengths)
        if do_compile:
            kn = min([k for k in self.static_lengths if k >= c])
            desc0, mask0 = pad_to_length(desc0, kn)
            desc1, mask1 = pad_to_length(desc1, kn)
            kpts0, _ = pad_to_length(kpts0, kn)
            kpts1, _ = pad_to_length(kpts1, kn)

        matchability = cosine_similarity(desc0, desc1)

        assert self.conf.pct_pruning >= 0 and self.conf.pct_pruning < 1
        if self.conf.pct_pruning > 0:
            ind0, ind1 = self.get_pruned_indices(matchability, self.conf.pct_pruning)

            matchability = gather(matchability, ind0)
            matchability = gather(matchability.mT, ind1).mT
            
            desc0 = gather(desc0, ind0)
            desc1 = gather(desc1, ind1)

            kpts0 = gather(kpts0, ind0)
            kpts1 = gather(kpts1, ind1)

        if self.conf.task == "object":
            bbox = data0["bbox"] # (B, 4)
            ind0, mask0 = self.get_prompted_indices(kpts0, bbox)

            matchability[:, 0] = torch.zeros_like(matchability[:, 0], device=matchability.device)
            desc0[:, 0] = torch.zeros_like(desc0[:, 0], device=desc0.device)
            kpts0[:, 0] = torch.zeros_like(kpts0[:, 0], device=kpts0.device)

            matchability = gather(matchability, ind0)
            desc0 = gather(desc0, ind0)
            kpts0 = gather(kpts0, ind0)

        # return matchability, kpts0, kpts1

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings

        kpts0 = normalize_keypoints(kpts0, intrinsic0)
        kpts1 = normalize_keypoints(kpts1, intrinsic1)

        encoding0 = self.posenc(kpts0).unsqueeze(-3)
        encoding1 = self.posenc(kpts1).unsqueeze(-3)
        # encoding0 = torch.zeros_like(encoding0, device=encoding0.device)
        # encoding0 = torch.zeros_like(encoding1, device=encoding1.device)

        for i in range(self.conf.n_layers):
            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1, match=matchability, mask0=mask0, mask1=mask1,
            )

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        desc0, desc1 = desc0.mean(1), desc1.mean(1)
        
        feat = torch.cat([desc0, desc1], 1)

        R = self.rotation_regressor(feat)
        R = rotation_matrix_from_ortho6d(R)
        t = self.translation_regressor(feat)
        # pose = self.regressor(feat)
        # R = rotation_matrix_from_ortho6d(pose[..., :6])
        # t = pose[..., 6:]

        return R, t
    
    def get_pruned_indices(self, match, pct_pruning):
        matching_scores0 = match.mean(-1)
        matching_scores1 = match.mean(-2)

        num_pruning0 = int(pct_pruning * matching_scores0.size(-1))
        num_pruning1 = int(pct_pruning * matching_scores1.size(-1))

        _, indices0 = matching_scores0.sort()
        _, indices1 = matching_scores1.sort()

        indices0 = indices0[:, num_pruning0:]
        indices1 = indices1[:, num_pruning1:]

        return indices0, indices1
    
    def get_prompted_indices(self, kpts, bbox):
        # kpts: (B, M, 2)
        # bbox: (B, 4) - (x, y, x, y)
        x, y = kpts[..., 0], kpts[..., 1]
        mask = (x >= bbox[:, 0].unsqueeze(-1)) & (x <= bbox[:, 2].unsqueeze(-1))
        mask &= (y >= bbox[:, 1].unsqueeze(-1)) & (y <= bbox[:, 3].unsqueeze(-1))
        mask_sorted, indices = mask.long().sort(descending=True)
        indices *= mask_sorted
        indices = indices[:, :mask_sorted.sum(-1).max()]
        mask_sorted = mask_sorted[:, :mask_sorted.sum(-1).max()]

        return indices, mask_sorted

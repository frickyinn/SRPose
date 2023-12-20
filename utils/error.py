import torch


def quat_degree_error(q1, q2):
    return 2 * torch.acos((q1 * q2).sum(1).abs()) * 180 / torch.pi


def rot_degree_error(R, Rgt):
    """
    Computes rotation degree error of residual rotation angle [radians]
    Input:
    R - estimated rotation matrix [B, 3, 3]
    Rgt - groundtruth rotation matrix [B, 3, 3]
    Output: degree error
    """

    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
    R_err = torch.acos(cosine)
    # loss = F.l1_loss(R_err, torch.zeros_like(R_err))

    return R_err

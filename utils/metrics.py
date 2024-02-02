import numpy as np
import torch
import sklearn
from scipy import spatial

from .pose_solver import EssentialMatrixSolver


def quat_degree_error(q1, q2):
    return 2 * torch.acos((q1 * q2).sum(1).abs()) * 180 / torch.pi


def rotation_angular_error(R, Rgt):
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


def translation_angular_error(t, tgt):
    cosine = torch.cosine_similarity(t, tgt)
    cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
    t_err = torch.acos(cosine)
    # t_err = t_err.rad2deg()
    # t_err = torch.minimum(t_err, 180-t_err)
    # t_err[tgt.norm(2, dim=1) < 0] = 0

    return t_err


def error_auc(errors, thresholds, mode):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'{mode}_auc@{t}': auc for t, auc in zip(thresholds, aucs)}


# def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
#     if len(kpts0) < 5:
#         return None
#     # normalize keypoints
#     kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
#     kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

#     # normalize ransac threshold
#     ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

#     # compute pose with cv2
#     E, mask = cv2.findEssentialMat(
#         kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
#     if E is None:
#         print("\nE is None while trying to recover pose.\n")
#         return None

#     # recover pose from E
#     best_num_inliers = 0
#     ret = None
#     for _E in np.split(E, len(E) / 3):
#         n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
#         if n > best_num_inliers:
#             ret = (R, t[:, 0], mask.ravel() > 0)
#             best_num_inliers = n

#     return ret


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def compute_pose_errors(pts0, pts1, K0, K1, T_0to1, cfg):
    solver = EssentialMatrixSolver(cfg)
    ret = solver.estimate_pose(pts0, pts1, {'K_color0': K0, 'K_color1': K1})

    if np.isnan(ret[0]).any():
        R_err = 180
        t_err = 180
        inliers = np.array([]).astype(bool)
    else:
        R, t, inliers = ret
        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
    
    return R_err, t_err, inliers


def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T

def reproj(K, R_est, t_est, R_gt, t_gt, pts):
    """
    reprojection error.
    :param K intrinsic matrix
    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    pixels_est = K.dot(pts_est.T)
    pixels_est = pixels_est.T
    pixels_gt = K.dot(pts_gt.T)
    pixels_gt = pixels_gt.T

    n = pts.shape[0]
    est = np.zeros((n, 2), dtype=np.float32);
    est[:, 0] = np.divide(pixels_est[:, 0], pixels_est[:, 2])
    est[:, 1] = np.divide(pixels_est[:, 1], pixels_est[:, 2])

    gt = np.zeros((n, 2), dtype=np.float32);
    gt[:, 0] = np.divide(pixels_gt[:, 0], pixels_gt[:, 2])
    gt[:, 1] = np.divide(pixels_gt[:, 1], pixels_gt[:, 2])

    e = np.linalg.norm(est - gt, axis=1).mean()
    return e

def add(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

def adi(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def compute_continuous_auc(metrics, thresholds):
    x_range = thresholds.max() - thresholds.min()
    results = np.array(metrics)
    accuracies = [(results <= t).sum() / len(results) for t in thresholds]
    auc = sklearn.metrics.auc(thresholds, accuracies) / x_range

    return auc

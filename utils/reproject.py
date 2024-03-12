import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


@dataclass
class Pose:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {'float': lambda v: f'{v:.6f}'}
        max_line_width = 1000
        q_str = np.array2string(self.q, formatter=formatter, max_line_width=max_line_width)[1:-1]
        t_str = np.array2string(self.t, formatter=formatter, max_line_width=max_line_width)[1:-1]
        return f'{self.image_name} {q_str} {t_str} {self.inliers}'
    

def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, 'w') as zip:
        for scene, poses in results_dict.items():
            poses_str = '\n'.join((str(pose) for pose in poses))
            zip.writestr(f'pose_{scene}.txt', poses_str.encode('utf-8'))



def project(pts: np.ndarray, K: np.ndarray, img_size: List[int] or Tuple[int] = None) -> np.ndarray:
    """Projects 3D points to image plane.

    Args:
        - pts [N, 3/4]: points in camera coordinates (homogeneous or non-homogeneous)
        - K [3, 3]: intrinsic matrix
        - img_size (width, height): optional, clamp projection to image borders
        Outputs:
        - uv [N, 2]: coordinates of projected points
    """

    assert len(pts.shape) == 2, 'incorrect number of dimensions'
    assert pts.shape[1] in [3, 4], 'invalid dimension size'
    assert K.shape == (3, 3), 'incorrect intrinsic shape'

    uv_h = (K @ pts[:, :3].T).T
    uv = uv_h[:, :2] / uv_h[:, -1:]

    if img_size is not None:
        uv[:, 0] = np.clip(uv[:, 0], 0, img_size[0])
        uv[:, 1] = np.clip(uv[:, 1], 0, img_size[1])

    return uv


def get_grid_multipleheight() -> np.ndarray:
    # create grid of points
    ar_grid_step = 0.3
    ar_grid_num_x = 7
    ar_grid_num_y = 4
    ar_grid_num_z = 7
    ar_grid_z_offset = 1.8
    ar_grid_y_offset = 0

    ar_grid_x_pos = np.arange(0, ar_grid_num_x)-(ar_grid_num_x-1)/2
    ar_grid_x_pos *= ar_grid_step

    ar_grid_y_pos = np.arange(0, ar_grid_num_y)-(ar_grid_num_y-1)/2
    ar_grid_y_pos *= ar_grid_step
    ar_grid_y_pos += ar_grid_y_offset

    ar_grid_z_pos = np.arange(0, ar_grid_num_z).astype(float)
    ar_grid_z_pos *= ar_grid_step
    ar_grid_z_pos += ar_grid_z_offset

    xx, yy, zz = np.meshgrid(ar_grid_x_pos, ar_grid_y_pos, ar_grid_z_pos)
    ones = np.ones(xx.shape[0]*xx.shape[1]*xx.shape[2])
    eye_coords = np.concatenate([c.reshape(-1, 1)
                                for c in (xx, yy, zz, ones)], axis=-1)
    return eye_coords


# global variable, avoids creating it again
eye_coords_glob = get_grid_multipleheight()


def reprojection_error(
        R_est: np.ndarray, t_est: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray, K: np.ndarray,
        W: int, H: int) -> float:
    eye_coords = eye_coords_glob

    # obtain ground-truth position of projected points
    uv_gt = project(eye_coords, K, (W, H))

    # residual transformation
    cam2w_est = np.eye(4)
    if not np.isnan(R_est).any():
        cam2w_est[:3, :3] = R_est
        cam2w_est[:3, -1] = t_est
    cam2w_gt = np.eye(4)
    cam2w_gt[:3, :3] = R_gt
    cam2w_gt[:3, -1] = t_gt

    # residual reprojection
    eyes_residual = (np.linalg.inv(cam2w_est) @ cam2w_gt @ eye_coords.T).T
    uv_pred = project(eyes_residual, K, (W, H))

    # get reprojection error
    repr_err = np.linalg.norm(uv_gt - uv_pred, ord=2, axis=1)
    mean_repr_err = float(repr_err.mean().item())
    return mean_repr_err

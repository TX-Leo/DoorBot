import numpy as np
import copy

from utils.lib_geo_trans import world2pixel
from utils_rgbd.lib_rgbd import CameraInfo

pts_3d = np.array([-0.16161697,0.01018764,0.9166841])

resize_ratio = 0.2

_cam_intrin = CameraInfo("config/cam_params_realsense_ours.json")
_cam_intrin_resized = copy.deepcopy(_cam_intrin)
_cam_intrin_resized.resize(resize_ratio)
intrin_mat_resized = _cam_intrin_resized.intrinsic_matrix(type="matrix")  # For the resized smaller image.

pts_2d_resized = world2pixel(
                pts_3d,
                T_cam_to_world=np.identity(4),
                camera_intrinsics=intrin_mat_resized).T
pts_2d_center = np.mean(pts_2d_resized, axis=0) / resize_ratio

print(f'pts_2d_resized:{pts_2d_resized}')
print(f'pts_2d_center:{pts_2d_center}')

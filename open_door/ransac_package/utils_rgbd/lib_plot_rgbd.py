'''
Plotting function for RGB, RGBD, Camera, PointCloud, etc.
'''

import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

if True:  # Add project root
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/../'
    sys.path.append(ROOT)

    from utils.lib_geo_trans import xyz_to_T, rot3x3_to_4x4, rot, rotx, roty, rotz
    from utils.lib_geo_trans import world2pixel, world2cam, cam2pixel


def to_ints(values):
    return tuple(int(v) for v in values)


def draw3dArrowOnImage(img_disp,
                       camera_intrinsics,
                       p0_xyz_in_world,
                       p1_xyz_in_world,
                       color="r",
                       line_width=2,
                       tipLength=0.3,
                       T_cam_to_world=np.identity(4),
                       distortion_coeffs=None,  # Not implemented.
                       ):
    '''
    Draw arrow onto image from p0 to p1.
    Arguments:
        p0_xyz_in_world {3-vec}: Starting point of the arrow.
        p1_xyz_in_world {3-vec}: End point of the arrow.
        camera_intrinsics {3x3 matrix}
        line_width {int}
        color: 
            Case 1: 'r' or 'g' or 'b'
            Case 2: A 3-vector of type uint8 that represents the color.
    Return:
        p0: (x0, y0), the starting point of the arrow.
        p1: (x1, y1), the ending point of the arrow.
    '''

    # -- Check input.
    assert(distortion_coeffs is None)  # TODO: Implement this.
    assert(isinstance(line_width, int))
    COLORS_DICT = {'b': [255, 0, 0], 'g': [0, 255, 0], 'r': [0, 0, 255]}
    IS_COLOR_GOOD = True
    if isinstance(color, str):
        if color in COLORS_DICT:
            color = COLORS_DICT[color]
        else:
            IS_COLOR_GOOD = False
    elif (isinstance(color, list) or isinstance(color, np.ndarray)) and len(color) == 3:
        pass  # Good
    else:
        IS_COLOR_GOOD = False
    if not IS_COLOR_GOOD:
        raise RuntimeError("Invalid color: " + color)

    # -- Project 3D point to 2D.
    def pt3d_to_pt2d(pt_3d):
        pt_2d = to_ints(world2pixel(
            pt_3d, T_cam_to_world, camera_intrinsics, distortion_coeffs))
        return pt_2d
    p0 = pt3d_to_pt2d(p0_xyz_in_world)
    p1 = pt3d_to_pt2d(p1_xyz_in_world)

    # -- Draw arrow.
    if isinstance(color, np.ndarray):
        color = color.tolist()
    if False:
        # Simplified arrow, only a dot and a line.
        cv2.line(img_disp, p0, p1, color, line_width)
        cv2.circle(img_disp, p0,
                   radius=line_width, color=color, thickness=-1)
    else:
        cv2.arrowedLine(
            img_disp, p0, p1, color=color,
            thickness=line_width, tipLength=tipLength)
    return p0, p1


def drawMaskFrom2dPoints(
    pts_2d,
    mask_shape,
    dilation_kernel_size=3,  # Dilate the mask image.
    dilation_times=0,
    distortion_coeffs=None,
):
    ''' Draw 2d points onto an empty mask.
    Arguments:
        pts_2d: 
            shape=(N, 2) or (2, N), np.float32. 
            Each point is at (x, y) position of the image.
        mask_shape {array}
    Return:
        mask {np.ndarray, np.uint8}: mask.shape==mask_shape.
            mask[i, j] is 0 or 255.
    '''
    if pts_2d.shape[0] != 2:  # (N, 2) --> (2, N)
        pts_2d = pts_2d.T
    rows, columns = mask_shape[0:2]

    # Convert pixel from float to int.
    xs, ys = pts_2d[0].astype(np.int16), pts_2d[1].astype(np.int16)

    # Remove pixels that are out of image.
    valid_x = np.logical_and(0 <= xs, xs < columns)
    valid_y = np.logical_and(0 <= ys, ys < rows)
    valid_xy = np.logical_and(valid_x, valid_y)
    xs = xs[valid_xy]
    ys = ys[valid_xy]

    # Draw mask.
    mask = np.zeros((rows, columns), np.uint8)
    mask[ys, xs] = 255
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    mask = cv2.dilate(src=mask, kernel=kernel,
                      iterations=dilation_times)
    return mask

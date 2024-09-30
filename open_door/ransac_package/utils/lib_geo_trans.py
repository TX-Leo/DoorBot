
'''
Geometric and camera related transformations
'''

import numpy as np
import copy
import cv2

''' =============================================================================== '''
''' Basic maths. '''


def form_T(R, p):
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3:4] = np.array(p).reshape((3, 1))
    return T


def get_Rp_from_T(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3:4]
    return (R, p)


def inv_R_p(R, p):
    T = form_T(R, p)
    T = np.linalg.inv(T)
    R_inv, p_inv = get_Rp_from_T(T)
    return R_inv, p_inv


def xyz_to_T(x=None, y=None, z=None):
    ''' Get 4x4 Transformation matrix from
        the translation (x, y, z)
    '''
    T = np.identity(4)
    data = [x, y, z]
    for i in range(3):
        if data[i] is not None:
            T[i, 3] = data[i]
    return T


def rot3x3_to_4x4(R):
    T = np.identity(4)
    T[0:3, 0:3] = R
    return T


def rot(axis, angle, matrix_len=4):
    R_vec = np.array(axis).astype(float)*angle
    R, _ = cv2.Rodrigues(R_vec)
    if matrix_len == 4:
        R = rot3x3_to_4x4(R)
    return R


def rotx(angle, matrix_len=4):
    return rot([1, 0, 0], angle, matrix_len)


def roty(angle, matrix_len=4):
    return rot([0, 1, 0], angle, matrix_len)


def rotz(angle, matrix_len=4):
    return rot([0, 0, 1], angle, matrix_len)


def euler2matrix(x, y, z, order='rxyz'):
    return rotx(x).dot(roty(y)).dot(rotz(z))


''' =============================================================================== '''
''' Camera related transformations between world/camera/image. '''


def distortPoint(x, y, distortion_coeffs=None):
    ''' Distort a point. 
    Arguments:
        x {float}, y {float}: Point's position on the camera normalized plane (z=1).
        distortion_coeffs {array}: 5 parameters; Radial/tangential model.
    Distortion direction: When points are from world to image.
        As mentioned in: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html,
        the distortation is considered to be happened
        when points are projected from the real world to the image.
    Return: 
        x_distort {float}, y_distort {float}.
    '''
    if distortion_coeffs is None:
        return x, y
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    d = distortion_coeffs
    k1, k2, p1, p2, k3 = d[0], d[1], d[2], d[3], d[4]
    x_distort = x * (1 + k1 * r2 + k2 * r4 + k3 * r6) + \
        2*p1*x*y + p2*(r2 + 2*x*x)
    y_distort = y * (1 + k1 * r2 + k2 * r4 + k3 * r6) + \
        p1*(r2 + 2*y*y) + 2*p2*x*y
    return x_distort, y_distort


def world2cam(pts_3d_world, T_cam_to_world):
    ''' Project points represented in world coordinate to image coordinate.
    Arguments:
        pts {np.ndarray}: 3xN.
        T_cam_to_world {np.ndarray}: 4x4.
    Return:
        pts_3d_camera {np.ndarray}: 3xN. Points in camera coordinate.
    '''
    # -- Check input.
    if type(pts_3d_world) == list:
        pts_3d_world = np.array(pts_3d_world)
    if len(pts_3d_world.shape) == 1:  # (3, ) --> (3, 1)
        pts_3d_world = pts_3d_world[:, np.newaxis]

    # -- Transform.
    # Change image to homogeneous coordinate.
    if pts_3d_world.shape[0] == 3:  # (3, N) --> (4, N)
        blanks = np.ones((1, pts_3d_world.shape[1]))
        pts_3d_world = np.vstack((pts_3d_world, blanks))
    pts_3d_camera = T_cam_to_world.dot(pts_3d_world)
    pts_3d_camera = pts_3d_camera[0:3, :]  # Remove homogeneous coordinate.
    return pts_3d_camera


def cam2pixel(pts_3d, camera_intrinsics, distortion_coeffs=None):
    ''' Project points represented in camera coordinate onto the image plane.
    Arguments:
        pts {np.ndarray}: 3xN.
        camera_intrinsics {np.ndarray}: 3x3.
    Return:
        image_points_xy {np.ndarray, np.float32}: 2xN.
    '''
    # -- Check input.
    if type(pts_3d) == list:
        pts_3d = np.array(pts_3d)
    if len(pts_3d.shape) == 1:  # (3, ) --> (3, 1)
        pts_3d = pts_3d[:, np.newaxis]

    # -- Transform
    # Transform to camera normalized plane (z=1)
    pts_3d = pts_3d/pts_3d[2, :]  # z=1

    # Distort point
    if distortion_coeffs is not None:
        for i in range(pts_3d.shape[1]):
            pts_3d[0, i], pts_3d[1, i] = distortPoint(
                pts_3d[0, i], pts_3d[1, i], distortion_coeffs)

    # Project to image plane
    image_points_xy = camera_intrinsics.dot(pts_3d)[0:2, :]
    return image_points_xy


def world2pixel(pts_3d_in_world, T_cam_to_world, camera_intrinsics, distortion_coeffs=None):
    ''' Combination of `world2cam` and `cam2pixel`.
    Arguments:
        pts {np.ndarray}: 3xN.
        T_cam_to_world {np.ndarray}: 4x4.
        camera_intrinsics {np.ndarray}: 3x3.
    Return:
        image_points_xy {np.ndarray, np.float32}: 2xN.
    '''
    # -- Check input.
    camera_intrinsics = np.array(camera_intrinsics)
    if camera_intrinsics.shape != (3, 3):
        raise RuntimeError("The camera_intrinsics needs to be a 3x3 matrix.")

    if isinstance(pts_3d_in_world, np.ndarray) and len(pts_3d_in_world.shape) == 2 \
            and pts_3d_in_world.shape[0] != 3:  # Nx3 --> 3xN
        pts_3d_in_world = pts_3d_in_world.T

    # -- Transform coordinate.
    image_points_xy = cam2pixel(
        world2cam(pts_3d_in_world, T_cam_to_world), camera_intrinsics, distortion_coeffs)
    return image_points_xy

''' =============================================================================== '''
''' Unit tests. '''


def test_basic_maths():
    R = euler2matrix(np.pi/2, 0, 0)
    R = rotz(np.pi/2)
    print(R)


if __name__ == "__main__":
    test_basic_maths()
    pass

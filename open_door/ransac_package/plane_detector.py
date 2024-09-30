#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d
import cv2
import matplotlib.pyplot as plt
import copy
import argparse
import os
import json

from utils.lib_io import read_yaml_file
from utils.lib_ransac import PlaneModel, RansacPlane
from utils.lib_geo_trans import world2pixel
from utils_rgbd.lib_rgbd import CameraInfo, resize_color_and_depth
from utils_rgbd.lib_open3d import wrap_open3d_point_cloud_with_my_functions
from utils_rgbd.lib_plot_rgbd import drawMaskFrom2dPoints, draw3dArrowOnImage
wrap_open3d_point_cloud_with_my_functions()

MAX_OF_MAX_PLANE_NUMBERS = 5


def subtract_points(points, indices_to_subtract):
    ''' Subtract points by indices and return a new sub cloud. '''
    all_indices = np.arange(len(points))
    rest_indices = np.setdiff1d(all_indices, indices_to_subtract)
    points = points[rest_indices]
    return points


class PlaneParam(object):
    ''' The parameters of the detected plane are stored in this class. '''

    def __init__(
            self, w, pts_3d_center, normal_vector, pts_2d_center, mask_color):
        self.w = w
        self.pts_3d_center = pts_3d_center
        self.normal_vector = normal_vector
        self.pts_2d_center = pts_2d_center
        self.mask_color = mask_color

    def resize_2d_params(self, ratio):
        self.pts_2d_center *= ratio

    def print_params(self, index=""):
        ''' Print the plane parameters. '''
        print("-----------------------------------")
        print("Plane {}: ".format(index))
        print("     weights: {}".format(self.w))
        print("     normal: {}".format(self.normal_vector))
        print("     3d center: {}".format(self.pts_3d_center))
        print("     2d center: {}".format(self.pts_2d_center))
        print("     mask color: {}".format(self.mask_color))
    
    def save_plane_params(self,save_path=''):
        data = {'weights': self.w.tolist(),
                'normal': self.normal_vector.tolist(),
                '3d_center': self.pts_3d_center.tolist(),
                '2d_center': self.pts_2d_center.tolist(),
                'mask_color': self.mask_color.tolist()
                }
        with open(save_path, 'w') as json_file:
            json.dump(data, json_file,indent=4)
        print("Saved plane parameters to {}".format(save_path))

class PlaneDetector(object):
    def __init__(self, config_file_path, camera_info_file_path):

        # -- Load config file.
        self._cfg = read_yaml_file(
            config_file_path, is_convert_dict_to_class=True)

        # -- Load camera intrinsics from file.
        self._cam_intrin = CameraInfo(camera_info_file_path)
        self._shape = self._cam_intrin.get_img_shape()

        # -- Algorithm settings.

        # Settings for reducing image size.
        self._cam_intrin_resized = copy.deepcopy(self._cam_intrin)
        self._cam_intrin_resized.resize(self._cfg.img_resize_ratio)
        self._shape_resized = self._cam_intrin_resized.get_img_shape()

        assert(self._cfg.max_number_of_planes <= MAX_OF_MAX_PLANE_NUMBERS)

        # -- Visualization settings.
        self._cmap = plt.get_cmap(self._cfg.visualization["color_map_name"])

    def detect_planes(self, depth_img, color_img=None):
        '''
        Arguments:
            depth_img {np.ndarry, np.uint16}:
                Undistorted depth image.
            color_img {None} or {np.ndarry, np.uint8, bgr, undistorted}:
                Color image is only for visualiation purpose.
                If None, color_img will be created as a black image.
        '''

        # -- Check input.
        if len(depth_img.shape) != 2:
            raise RuntimeError("Depth image should have 2 channels.")
        if color_img is None:  # Use black image instead.
            r, c = depth_img.shape[0:2]
            color_img = np.zeros(shape=(r, c, 3), dtype=np.uint8)

        # -- Resize image.
        color_img_resized, depth_img_resized = resize_color_and_depth(
            color_img, depth_img, self._cfg.img_resize_ratio)

        # -- Compute point cloud.
        pcd = self._create_point_cloud(color_img_resized, depth_img_resized)
        if self._cfg.debug["draw_3d_point_cloud"]:
            pcd.draw()
        points = pcd.get_xyzs()
        # points.shape=(N, 3). Each row is a point's 3d position of (x, y, z).

        # -- Detect plane one by one until there is no plane.
        planes = []
        for i in range(self._cfg.max_number_of_planes):
            print("-------------------------")
            print("Start detecting {}th plane ...".format(i))

            # Detect plane by RANSAC.
            is_succeed, plane_weights, plane_pts_indices = \
                self._detect_plane_by_RANSAC(points)
            if not is_succeed:
                break

            # Store plane result.
            plane_points = points[plane_pts_indices]
            planes.append(self._Plane(plane_weights, plane_points))

            # Use the remaining point cloud to detect next plane.
            points = subtract_points(points, plane_pts_indices)
        print("-------------------------")
        print("Plane detection completes. Detect {} planes.".format(len(planes)))

        # -- Process planes to obtain desired plane parameters.
        list_plane_params, planes_mask, planes_img_viz = \
            self._compute_planes_info(planes, color_img)

        # -- Return.
        return list_plane_params, planes_mask, planes_img_viz, pcd

    class _Plane(object):
        def __init__(self, plane_weights, plane_points):
            self.weights = plane_weights
            self.points = plane_points

    def _create_point_cloud(self, color_img_resized, depth_img_resized):
        ''' Create point cloud from color and depth image.
        Return:
            pcd {open3d.geometry.PointCloud}
        '''

        # rgbd_image = open3d.create_rgbd_image_from_color_and_depth( # for old version
        rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth( # for new version
            # color=open3d.Image( # for old version
            color=open3d.geometry.Image( # for new version
                cv2.cvtColor(color_img_resized, cv2.COLOR_BGR2RGB)),
            # depth=open3d.Image(depth_img_resized), # for old version
            depth=open3d.geometry.Image(depth_img_resized), # for new version
            depth_scale=1.0/self._cfg.depth_unit,
            depth_trunc=self._cfg.depth_trunc,
            convert_rgb_to_intensity=False)

        cam_intrin = self._cam_intrin_resized.to_open3d_format()
        # pcd = open3d.create_point_cloud_from_rgbd_image( # for old version
        pcd = open3d.geometry.PointCloud.create_from_rgbd_image( # for new version
            rgbd_image,
            cam_intrin)

        if self._cfg.cloud_downsample_voxel_size > 0:
            pcd = open3d.geometry.voxel_down_sample(
                pcd, voxel_size=self._cfg.cloud_downsample_voxel_size)

        return pcd

    def _detect_plane_by_RANSAC(self, points):
        ''' Use RANSAC to detect plane from point pcd.
        The plane weights(parameters) w means:
            w[0] + w[1]*x + w[2]*y + w[3]*z = 0
        Arguments:
            points {np.ndarray}: (N, 3).
        Return:
            is_succeed {bool}: Is plane detected successfully.

        '''
        FAILURE_RETURN = False, None, None
        cfg = self._cfg.RANSAC_config

        print("\nRANSAC starts: Source points = {}".format(len(points)))
        ransac = RansacPlane()
        is_succeed, plane_weights, plane_pts_indices = ransac.fit(
            points,
            model=PlaneModel(),
            n_pts_fit_model=3,
            n_min_pts_inlier=cfg["min_points"],
            max_iter=cfg["iterations"],
            dist_thresh=cfg["dist_thresh"],
            is_print_res=cfg["is_print_res"],
        )

        if not is_succeed:
            print("RANSAC Failed.")
            return FAILURE_RETURN
        print("RANSAC succeeds: Points of plane = {}".format(
            plane_pts_indices.size))

        # Let the plane norm pointing to the camera.
        #       which means that the norm's z component should be negative.
        if plane_weights[-1] > 0:
            plane_weights *= -1
        return is_succeed, plane_weights, plane_pts_indices

    def _compute_planes_info(self, planes, color_img):
        '''
        Arguments:
            planes {list of `class _Plane`}
        Returns:
            list_plane_params {list of `class PlaneParam`}
            planes_mask {image}: Mask of the detected planes.
                Each plane corresponds to one color region of the mask.
                This is a colored mask, with 3 channels and np.uint8 datatype.
            planes_img_viz {image}: image visualization of the detected planes.
                For each plane region, a color is superposed onto the origin image.
                Besides, an arrow is drawn to indicate the plane direction.
        '''

        shape, shape_resized = self._shape, self._shape_resized

        intrin_mat = self._cam_intrin.intrinsic_matrix(type="matrix")
        intrin_mat_resized = self._cam_intrin_resized.intrinsic_matrix(
            type="matrix")  # For the resized smaller image.
        resize_ratio = self._cfg.img_resize_ratio
        cfg_viz = self._cfg.visualization

        # -- Initialize the output variables.
        merged_masks = np.zeros(
            (shape_resized[0], shape_resized[1], 3), np.uint8)
        planes_img_viz = color_img.copy()
        list_plane_params = []

        # -- Process each plane.
        for i, plane in enumerate(planes):
            w, pts_3d = plane.weights, plane.points

            # Project 3d points to 2d by using
            # the resized intrinsics,
            # so the created mask is small, and costs less time.
            pts_2d_resized = world2pixel(
                pts_3d,
                T_cam_to_world=np.identity(4),
                camera_intrinsics=intrin_mat_resized).T
            mask_resized = drawMaskFrom2dPoints(
                pts_2d_resized,
                shape_resized[:2],
                dilation_kernel_size=3,
                dilation_times=1)
            mask_resized = mask_resized > 0
            color = self._get_ith_color(i)
            merged_masks[mask_resized] = color
            # pts_3d.shape=(N, 2)
            # pts_2d_resized.shape=(N, 2)

            # -- Compute plane parameters:
            #   3d normal, 3d center, 2d center, mask color.
            normal_vector = w[1:]
            pts_3d_center = np.mean(pts_3d, axis=0)
            pts_2d_center = np.mean(pts_2d_resized, axis=0) / resize_ratio
            plane_param = PlaneParam(
                w, pts_3d_center, normal_vector, pts_2d_center, color)
            list_plane_params.append(plane_param)

            # -- Draw arrow on `planes_img_viz`.
            arrow_p1 = pts_3d_center
            arrow_p2 = pts_3d_center + normal_vector * \
                cfg_viz["arrow_length"]

            p0, p1 = draw3dArrowOnImage(
                planes_img_viz,
                intrin_mat,
                arrow_p1, arrow_p2,
                color,
                cfg_viz["arrow_linewidth"], cfg_viz["arrow_tip_length"],
                T_cam_to_world=np.identity(4),
            )

            # Draw a number near the arrow.
            p2 = calc_opposite_point(p0, p1, length=20.0)
            cv2.putText(planes_img_viz,
                        text=str(i+1),
                        org=p2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=color.tolist(), thickness=2)

        # Resize mask back to origin image size.
        planes_mask = cv2.resize(
            merged_masks, None,
            fx=1.0/resize_ratio, fy=1.0/resize_ratio,
            interpolation=cv2.INTER_NEAREST)

        return list_plane_params, planes_mask, planes_img_viz

    def _get_ith_color(self, i):
        color_tuple_float = self._cmap(
            float(i)/MAX_OF_MAX_PLANE_NUMBERS)[:3]
        # color_tuple_uint8 = tuple(int(c*255) for c in color_tuple_float)
        color_array_uint8 = (
            np.array(color_tuple_float)*255).astype(np.uint8)
        return color_array_uint8


def calc_opposite_point(p0, p1, length=5.0, to_int=True):
    ''' p0 and p1 are two points.
        Create a point p2 
        so that the vector (p2, p0) and (p0, p1) are at the same line.
        length is the length of p2~p0.
    '''
    x0, y0 = p0
    x1, y1 = p1
    theta = np.arctan2(y1 - y0, x1 - x0)
    x2 = x0 - length * np.cos(theta)
    y2 = y0 - length * np.sin(theta)
    if to_int:
        x2, y2 = int(x2), int(y2)
    return (x2, y2)


def plane_detector(rgb_img_path='images/rgb_img.png',d_img_path='images/d_img.png',config_file_path="config/plane_detector_config_ours.yaml",camera_info_file_path="config/cam_params_realsense_ours.json",vis=False):
    # -- Read color image and depth images
    image_dir = os.path.dirname(rgb_img_path)+'/ransac'
    rgb_img = cv2.imread(rgb_img_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    d_img = cv2.imread(d_img_path, cv2.IMREAD_UNCHANGED)

    # -- create a detector
    detector = PlaneDetector(config_file_path, camera_info_file_path)

    # -- Detect planes.
    list_plane_params, planes_mask, planes_img_viz, pcd = detector.detect_planes(
        d_img, rgb_img)

    # -- Print result. (set the max_number_of_planes=1 in config file before)
    for i, plane_param in enumerate(list_plane_params):
        plane_param.print_params(index=i+1)
        plane_param.save_plane_params(save_path=f'{image_dir}/ransac_result.json')

    # -- Plot result.
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(planes_mask)
        plt.title("Planes mask.")
        plt.subplot(1, 2, 2)
        plt.imshow(planes_img_viz)
        plt.title("Planes normals.")
        plt.savefig(f'{image_dir}/plane.png')
    if vis:
        plt.show()
    
    ## 3d show
    # for i, plane_param in enumerate(list_plane_params):
    #     arrow_p1 = plane_param.pts_3d_center
    #     arrow_p2 = plane_param.pts_3d_center + plane_param.normal_vector * 0.2
    #     point_A = np.array([x1, y1, z1])
    #     point_B = np.array([x2, y2, z2])
    #     line = o3d.geometry.LineSet()
    #     line.points = o3d.utility.Vector3dVector([point_A, point_B])
    #     line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # red
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()
    #     vis.add_geometry(pcd)
    #     vis.add_geometry(line)
    #     vis.get_render_option().point_size = 1.0
    #     vis.run()
    #     vis.destroy_window()

    normal = list_plane_params[0].normal_vector.tolist()
    return normal

def main(args):
    plane_detector(rgb_img_path=args.rgb_img_path,
                   d_img_path=args.d_img_path,
                   config_file_path = args.config_file_path,
                   camera_info_file_path = args.camera_info_file_path,
                   vis=args.vis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-rgb","--rgb_img_path",default='images/rgb.png',help="Input rgb image path.")
    parser.add_argument("-d","--d_img_path",default='images/d.png',help="Input d image path.")
    parser.add_argument("-cfg","--config_file_path",default="config/plane_detector_config_ours.yaml")
    parser.add_argument("-camera","--camera_info_file_path",default="config/cam_params_realsense_ours.json")
    parser.add_argument("-v","--vis",default=False,action='store_true',help="if vis.")
    main(parser.parse_args())
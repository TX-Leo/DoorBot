#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d
import numpy as np
import copy


def wrap_open3d_point_cloud_with_my_functions():
    ''' 
    After calling this function, 
        all `open3d.open3d.geometry.PointCloud` instances (whether created or not)
        will get extra functions added by this function. 
    Reference: https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
    '''
    t = open3d.open3d.geometry.PointCloud

    def clear(self):
        self.points = open3d.Vector3dVector(np.ndarray((0, 0)))
        self.colors = open3d.Vector3dVector(np.ndarray((0, 0)))
    t.clear = clear

    def copy_to(self, dst):
        dst.points = copy.deepcopy(self.points)
        dst.colors = copy.deepcopy(self.colors)
    t.copy_to = copy_to

    def size(self):
        ''' Return number of points in the point cloud. '''
        return np.asarray(self.points).shape[0]
    t.size = size

    def get_xyzs(self):
        ''' Get points' xyz values.
            Return Nx3 np.ndarray.
        '''
        return np.asarray(self.points)
    t.get_xyzs = get_xyzs

    def get_colors(self):
        ''' Get points' color values.
            Return Nx3 np.ndarray.
        '''
        return np.asarray(self.colors)
    t.get_xyzs = get_xyzs

    def get_xyzs_colors(self):
        return np.asarray(self.points), np.asarray(self.colors)
    t.get_xyzs_colors = get_xyzs_colors

    def draw(self):
        open3d.visualization.draw_geometries([self])
    t.draw = draw

    def select_points(self, indices):
        ''' Select points by indices and return a new sub cloud. '''
        points = np.asarray(self.points)[indices]
        colors = np.asarray(self.colors)[indices]
        sub_cloud = open3d.PointCloud()
        sub_cloud.points = open3d.geometry.Vector3dVector(points)
        sub_cloud.colors = open3d.geometry.Vector3dVector(colors)
        return sub_cloud
    t.select_points = select_points

    def subtract_points(self, indices_to_subtract):
        ''' Subtract points by indices and return a new sub cloud. '''
        num_points = np.asarray(self.points).shape[0]
        all_indices = np.range(num_points)
        rest_indices = np.setdiff1d(all_indices, indices_to_subtract)
        rest_pcd = self.select_points(rest_indices)
        return rest_pcd
    t.subtract_points = subtract_points


def test_wrap_open3d_point_cloud_with_my_functions():
    import os
    ROOT = os.path.join(os.path.dirname(__file__))+"/../"
    filename = ROOT + "data/pcd/bottle.pcd"
    cloud = open3d.io.read_point_cloud(filename)
    wrap_open3d_point_cloud_with_my_functions()
    print(cloud.size())
    cloud.draw()

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

class PointCloud(object):
    def __init__(self,pinhole_intrinsic,extrinsic,depth_scale):
        self.depth_scale = depth_scale
        self.pinhole_intrinsic = pinhole_intrinsic
        self.extrinsic = extrinsic

        self.pcd = None
    
    def save(self,pcd_path,if_p=False):
        o3d.io.write_point_cloud(pcd_path,self.pcd)
        if if_p:
            print(f'point cloud saved to {pcd_path}')
    
    def load(self,pcd_path):
        self.pcd = o3d.io.read_point_cloud(pcd_path)

    def downsample_pc(self,voxel_size=0.01,if_p=False):
        self.pcd.estimate_normals() 
        pcd_downsampled = self.pcd.voxel_down_sample(voxel_size=voxel_size) 
        if if_p:
            print(f'[points num before] {len(self.pcd.points)}')
            print(f'[points num after] {len(pcd_downsampled.points)}')
        return pcd_downsampled 
    
    def upsample_pc(self,upsampling_factor=2,if_p=False):
        """Approximates upsampling by duplicating points with slight offsets."""
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)
        new_points = []
        new_colors = []
        for i in range(upsampling_factor):
            for j in range(upsampling_factor):
                offset = np.array([i*0.001, j*0.001, 0])  # Adjust offset as needed
                new_points.append(points + offset)
                new_colors.append(colors)

        pcd_upsampled = o3d.geometry.PointCloud()
        pcd_upsampled.points = o3d.utility.Vector3dVector(np.concatenate(new_points, axis=0))
        pcd_upsampled.colors = o3d.utility.Vector3dVector(np.concatenate(new_colors, axis=0))
        if if_p:
            print(f'[points num before] {len(self.pcd.points)}')
            print(f'[points num after] {len(pcd_upsampled.points)}')

        return pcd_upsampled

    def gen_pc_from_d(self,d_img,pcd_path=None):
        # Create Open3D depth image
        self.o3d_depth = o3d.geometry.Image(d_img)

        # Create Point Cloud directly from depth
        self.pcd = o3d.geometry.PointCloud.create_from_depth_image(self.o3d_depth,intrinsic=self.pinhole_intrinsic,extrinsic=self.extrinsic,depth_scale=self.depth_scale) # depth_trunc=1000.0, # Optional: Set a maximum depth for clipping

        # Save
        if pcd_path:
            self.save(pcd_path)

        return self.pcd

    def gen_pc_from_rgbd(self,rgb_img,d_img,pcd_path=None):
        # Open3D Images
        self.o3d_color = o3d.geometry.Image(rgb_img)
        self.o3d_depth = o3d.geometry.Image(d_img)

        # rgb and d to rgbd
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(self.o3d_color,self.o3d_depth,depth_scale=self.depth_scale,convert_rgb_to_intensity=False)

        # rgbd to pcd
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd,intrinsic=self.pinhole_intrinsic,extrinsic=self.extrinsic)
        
        # save
        if pcd_path:
            self.save(pcd_path)

        return self.pcd
    
    def gen_pc_from_rgbd_img(self, rgb_img_path, d_img_path, pcd_path=None):
        color_raw = Image.open(rgb_img_path)
        if not color_raw.mode == "RGB":
            color_raw = color_raw.convert("RGB") 
        rgb_img = np.array(color_raw)
        
        depth_raw = Image.open(d_img_path)
        depth_raw = depth_raw.convert("I")  # Assuming your depth is unsigned int
        d_img = np.array(depth_raw).astype(np.uint16)

        self.pcd = self.gen_pc_from_rgbd(rgb_img,d_img,pcd_path)
        return self.pcd

    def gen_pc_from_d_img(self, d_img_path, pcd_path=None):
        depth_raw = Image.open(d_img_path)
        depth_raw = depth_raw.convert("I")  # Assuming your depth is unsigned int
        d_img = np.array(depth_raw).astype(np.uint16)

        self.pcd = self.gen_pc_from_d(d_img,pcd_path)
        return self.pcd

    def vis_pcd(self,img_path='pc.png',vis_parts=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Visualizer", width=1500, height=1500)
        vis.add_geometry(self.pcd)
        if vis_parts:
            for i in vis_parts:
                vis.add_geometry(i)

        render_opt = vis.get_render_option()
        render_opt.point_size = 1.0 

        vis.run()  # This will block until the window is closed

        if img_path:
            vis.capture_screen_image(img_path, do_render=True)
            print(f"Screenshot saved to {img_path}")

        vis.destroy_window()

    def vis_point_in_pc(self,point,img_path='pc.png',radius=0.005,color=[1,0,0]):
        # Create a sphere geometry to represent the point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)  # Move the sphere to the desired point location
        sphere.paint_uniform_color(color) # Set the color of the sphere

        self.vis_pcd(img_path,vis_parts=[sphere])
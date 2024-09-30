import time
import keyboard
import json
import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

from utils.lib_io import *
from utils.lib_pc import *

class CamIntrinsic(object):
    def __init__(self,intrinsic):
        self.fx = intrinsic[0]
        self.fy = intrinsic[4]
        self.cx = intrinsic[6]
        self.cy = intrinsic[7]
        self.intrinsic_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def __str__(self):
        return f"CamIntrinsic(\n  fx={self.fx},\n  fy={self.fy},\n  cx={self.cx},\n  cy={self.cy},\n  intrinsic_matrix=\n{self.intrinsic_matrix}\n)"

class Camera(object):
    def __init__(self,width=1280,height=720,intrinsic_matrix=None,extrinsic=None,depth_scale=0.001,fps=30):
        self.width = width
        self.height = height
        self.intrinsic = CamIntrinsic(intrinsic_matrix)
        self.pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.intrinsic.fx, self.intrinsic.fy, self.intrinsic.cx, self.intrinsic.cy)
        self.extrinsic = extrinsic
        self.depth_scale = depth_scale
        self.pc = PointCloud(self.pinhole_intrinsic,self.extrinsic,1/self.depth_scale)
        self.connect(fps)

    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_cam.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.width,cfg.height,cfg.intrinsic_matrix,cfg.extrinsic,cfg.depth_scale,cfg.fps)

    def __str__(self):
        return f"RealSense(\n  width={self.width},\n  height={self.height},\n  {self.intrinsic.__str__()},\n  depth_scale={self.depth_scale}\n)"

    def connect(self,fps=30):
        print('==========\nCamera Connecting...')
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_device('238122071696')
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        print('Camera Connected\n==========')

    def disconnect(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def init_intrinsic(self):
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        print(intrinsics)
        print("intrinsics:")
        print(f"Width: {intrinsics.width}")
        print(f"Height: {intrinsics.height}")
        print(f"FX: {intrinsics.fx}")
        print(f"FY: {intrinsics.fy}")
        print(f"CX: {intrinsics.ppx}")
        print(f"CY: {intrinsics.ppy}")
        print("distortion coefficient:")
        print(f"K1: {intrinsics.coeffs[0]}")
        print(f"K2: {intrinsics.coeffs[1]}")
        print(f"P1: {intrinsics.coeffs[2]}")
        print(f"P2: {intrinsics.coeffs[3]}")
        print(f"K3: {intrinsics.coeffs[4]}")
        return CamIntrinsic([intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy])

    def init_depth_scale(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        return depth_scale

    def capture_rgb(self,rgb_save_path=None):
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_color_frame = aligned_frames.get_color_frame()
        rgb_img = np.asanyarray(aligned_color_frame.get_data())
        if rgb_save_path is not None:
            cv2.imwrite(rgb_save_path,rgb_img)
        return rgb_img

    def capture_d(self,d_save_path=None):
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        d_img = np.asanyarray(aligned_depth_frame.get_data())
        if d_save_path is not None:
            cv2.imwrite(d_save_path,d_img)
        return d_img

    def capture_rgbd(self,rgb_save_path=None,d_save_path=None):
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        d_img = np.asanyarray(aligned_depth_frame.get_data())
        rgb_img = np.asanyarray(aligned_color_frame.get_data())
        if rgb_save_path is not None:
            cv2.imwrite(rgb_save_path,rgb_img)
        if d_save_path is not None:
            cv2.imwrite(d_save_path,d_img)
            # np.save(d_save_path, d_img)
        return rgb_img,d_img

    def capture_video(self,duration,fps,save_path=None):
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(save_path, fourcc, fps, (self.width, self.height))
        # Record for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert depth and color frames to OpenCV images
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # # Write combined depth and color image (optional)
            # combined_image = np.hstack((depth_image, color_image))
            # out.write(combined_image)
            # Write only color image
            out.write(color_image)
        out.release()

    def get_depth_gaussian(self,d_img,x_min,x_max,y_min,y_max):
        if isinstance(d_img, str):
            d_img = cv2.imread(d_img, cv2.IMREAD_UNCHANGED)
        
        # Extract the region of interest
        roi = d_img[y_min:y_max, x_min:x_max]

        # filter the region of interest (remove all zero)
        roi = roi[roi != 0]

        # print(f'roi: {roi}')
        mean = np.mean(roi)
        # print(f'mean: {mean}')
        std = np.std(roi)
        # print(f'std: {std}')
        threshold = std
        # print(f'threshold: {threshold}')

        outliers = np.array([x for x in roi if mean - x > threshold]) # handle value
        depth = np.mean(outliers)
        # print(f"Outliers: {outliers}")
        # print(f"depth: {depth}")
        
        return depth

    def get_handle_depth(self,x,y,d_img,orientation=None,radius=40):
        if isinstance(d_img, str):
            d_img = cv2.imread(d_img, cv2.IMREAD_UNCHANGED)
        x,y = int(x),int(y)
        if orientation == 'horizontal':
            x_min = x
            x_max = x+1
            y_min = y-radius
            y_max = y+radius
        elif orientation == 'vertical':
            x_min = x-radius
            x_max = x+radius
            y_min = y
            y_max = y+1
        handle_depth = self.get_depth_gaussian(d_img,x_min,x_max,y_min,y_max)
        return handle_depth

    def get_depth_point(self,x,y,d_img):
        if isinstance(d_img, str):
            d_img = cv2.imread(d_img, cv2.IMREAD_UNCHANGED)

        height, width = d_img.shape[:2]  # Get image height and width
        x = int(x)
        y = int(y)

        if 0 <= x < width and 0 <= y < height: 
            depth = d_img[y, x]
            return depth
        else:
            return None  # Or handle out-of-bounds case differently

    def get_depth_roi(self, u, v, d_img, radius=15, depth_threshold=0.05, valid_ratio_threshold=0.50):
        if isinstance(d_img, str):
            d_img = cv2.imread(d_img, cv2.IMREAD_UNCHANGED)

        # 1. Extract Region of Interest (ROI)
        u, v = int(u), int(v)  # Ensure integer indices
        height, width = d_img.shape[:2]
        u_min, u_max = max(0, u - radius), min(width - 1, u + radius)
        v_min, v_max = max(0, v - radius), min(height - 1, v + radius)
        depth_roi = d_img[v_min:v_max+1, u_min:u_max+1]

        # 2. Filter for Valid Depths
        center_depth = np.mean(depth_roi[depth_roi != 0]) # center_depth = depth_roi[radius, radius]
        valid_depth_mask = (np.abs(depth_roi - center_depth) * self.depth_scale <= depth_threshold) & (depth_roi != 0)
        
        # 3. Check for Sufficient Valid Data
        valid_ratio = np.sum(valid_depth_mask) / np.count_nonzero(depth_roi)
        if valid_ratio < valid_ratio_threshold:
            print(f"ERROR: Not enough valid depth values around the point. Ratio: {valid_ratio:.2f}")
            return None 
        
        # 4. Calculate Average Depth 
        average_depth = np.mean(depth_roi[valid_depth_mask])

        return average_depth
    
    def xy_depth_2_xyz(self,u,v,depth):
        fx = self.intrinsic.fx
        fy = self.intrinsic.fy
        cx = self.intrinsic.cx
        cy = self.intrinsic.cy
        x = (u - cx) * depth * self.depth_scale / fx
        y = (v - cy) * depth * self.depth_scale / fy
        z = depth * self.depth_scale
        return x, y, z

    def gen_pc_rs(self,pcd_path="pc.pcd"):
        ## need to wait for few seconds after starting the realsense, or else the visualization image will be wired!!!
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        d_img = np.asanyarray(aligned_depth_frame.get_data())
        rgb_img = np.asanyarray(aligned_color_frame.get_data())
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # Convert to BGR

        pcd = self.pc.gen_pc_from_rgbd(rgb_img,d_img,pcd_path)
       
        return pcd

    def vis_pc_realtime(self,pcd_path="pc.pcd",img_path='pc.png'):
        pcd = self.gen_pc_rs()
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Visualizer",width=1500, height=1500)
        vis.add_geometry(pcd)
        render_opt = vis.get_render_option()
        render_opt.point_size = 1.0

        while True:
            # break
            pcd_new = self.gen_pc_rs()
            pcd.points = pcd_new.points
            pcd.colors = pcd_new.colors
            vis.update_geometry(pcd)
            if vis.poll_events():
                vis.update_renderer()
            else: # Close window manually
                break
        
        if pcd_path:
            o3d.io.write_point_cloud(pcd_path, pcd)  # Save as .ply (you can change the format)
            print(f'point cloud saved to {pcd_path}')
        if img_path:
            vis.capture_screen_image(img_path, do_render=True)
            print(f"Screenshot saved to {img_path}")

        vis.destroy_window()

    def check_rs_resolution(self):
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        for profile in depth_sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.depth:
                width, height = profile.as_video_stream_profile().width(), profile.as_video_stream_profile().height()
                print(f"Depth Stream Resolution: {width} x {height}")
        for profile in device.query_sensors()[1].get_stream_profiles():
            if profile.stream_type() == rs.stream.color:
                width, height = profile.as_video_stream_profile().width(), profile.as_video_stream_profile().height()
                print(f"Color Stream Resolution: {width} x {height}")

    def get_serial_num(self):
        devices = rs.context().query_devices()
        for dev in devices:
            serial_number = dev.get_info(rs.camera_info.serial_number)
            print(f"Device: {serial_number}")

    def display_and_record(self):
        cv2.namedWindow('RealSense RGB', cv2.WINDOW_AUTOSIZE)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        recording = False
        frame_count = 0
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow('RealSense RGB', color_image)
                # Record if 'r' is pressed
                if keyboard.is_pressed('r'):
                    out = cv2.VideoWriter('realsense_clip.avi', fourcc, 30.0, (1280, 720))
                    recording = True
                    print("Recording started.")
                # Pause recording if 'p' is pressed
                if keyboard.is_pressed('p'):
                    recording = False
                    print("Recording paused.")
                # Save frame to video if recording is enabled
                if recording:
                    out.write(color_image)
                    frame_count += 1
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            out.release()
            print(f"Recording stopped. {frame_count} frames recorded.")

    def rgbd_viewer(self):
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)

        finally:

            # Stop streaming
            self.pipeline.stop()

if __name__ == "__main__":
    camera = Camera.init_from_yaml(cfg_path='cfg/cfg_cam.yaml')
    print(camera)
    
    ## rgbd viewer
    camera.rgbd_viewer()

    ## capture rgbd
    # camera.capture_rgbd(rgb_save_path='test_rgb.png',d_save_path='test_d.npy')

    ## display_and_record
    # camera.display_and_record()
    
    ## visualize pc realtime
    # time.sleep(5)
    # camera.vis_pc_realtime(pcd_path='pc.pcd',img_path='pc.png')
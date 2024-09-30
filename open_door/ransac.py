'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-23 19:00:25
Version: v1
File: 
Brief: 
'''
import os
import json
class RANSAC():
    def __init__(self,cfg_ransac='cfg/cfg_ransac.yaml',cfg_cam='cfg/cfg_cam.yaml',vis=False):
        self.config_file_path=cfg_ransac
        self.camera_info_file_path=cfg_cam
        self.vis=vis

    def get_normal(self,rgb_img_path,d_img_path):
        from ransac_package.plane_detector import plane_detector
        normal = plane_detector(rgb_img_path,d_img_path,self.config_file_path,self.camera_info_file_path,self.vis)
        return normal

    def get_normal_server(self,rgb_img_path,d_img_path,server,remote_python_path,remote_root_dir,remote_img_dir):
        local_rgb_img_path = rgb_img_path
        remote_rgb_img_path = f'{remote_img_dir}/{os.path.basename(local_rgb_img_path)}'
        local_d_img_path = d_img_path
        remote_d_img_path = f'{remote_img_dir}/{os.path.basename(local_d_img_path)}'
        local_config_file_path = self.config_file_path
        remote_config_file_path = f'{remote_img_dir}/ransac/{os.path.basename(local_config_file_path)}'
        local_camera_info_file_path = self.camera_info_file_path
        remote_camera_info_file_path = f'{remote_img_dir}/ransac/{os.path.basename(local_camera_info_file_path)}'

        # transfer the input files to the server
        server.exec_cmd(f'mkdir -p {remote_img_dir}/ransac/')
        server.transfer_file_local2remote(local_rgb_img_path,remote_rgb_img_path)
        server.transfer_file_local2remote(local_d_img_path,remote_d_img_path)
        server.transfer_file_local2remote(local_config_file_path,remote_config_file_path)
        server.transfer_file_local2remote(local_camera_info_file_path,remote_camera_info_file_path)

        # ransac
        remote_ransac_script_dir = f'{remote_root_dir}/ransac_package/'
        remote_ransac_script_path = f'plane_detector.py'
        ransac_cmd = f'cd {remote_ransac_script_dir}; {remote_python_path} {remote_ransac_script_path} -rgb {remote_rgb_img_path} -d {remote_d_img_path} -cfg {remote_config_file_path} -camera {remote_camera_info_file_path}'
        server.exec_cmd(ransac_cmd)

        # transfer the output dir to the server
        server.transfer_folder_remote2local(f'{remote_img_dir}/ransac/', f'{os.path.dirname(local_rgb_img_path)}/ransac/')

        # open ransac_result.json to get rx and ry and rz
        with open(f'{os.path.dirname(local_rgb_img_path)}/ransac/ransac_result.json','r') as f:
            data = json.load(f)
            normal = data['normal']
            weights = data['weights']
            _3d_center = data['3d_center']
            _2d_center = data['2d_center']
            mask_color = data['mask_color']
        return normal,weights,_3d_center,_2d_center,mask_color

if __name__ == "__main__":  
    ransac = RANSAC(cfg_ransac='cfg/cfg_ransac.yaml',cfg_cam='cfg/cfg_cam.yaml',vis=False)
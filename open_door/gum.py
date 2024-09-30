'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-20 12:27:30
Version: v1
File: 
Brief: 
'''
import os
import json
import time

from utils.lib_io import *

class GUM(object):
    def __init__(self,img_w=640,img_h=640,model_path='checkpoints/gum.pth',device='cuda:0'):
        self.img_w = img_w
        self.img_h = img_h
        self.model_path = model_path
        self.device = device
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_gum.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.img_w,cfg.img_h,cfg.model_path,cfg.device)

    def get_dxdyR(self,image_path='',mask_path='',root_dir=''):
        from gum_package.get_dxdyR import get_dxdyR
        dx,dy,R = get_dxdyR(image_path,mask_path,self.model_path,self.device,root_dir)
        return dx,dy,R
    
    def get_dxdyR_server(self,image_path,mask_path,server,remote_python_path,remote_root_dir,remote_img_dir):
        start_time = time.time()

        local_rgb_img_path = image_path
        remote_rgb_img_path = f'{remote_img_dir}/{os.path.basename(local_rgb_img_path)}'
        local_mask_path = mask_path
        remote_mask_path = f'{remote_img_dir}/{os.path.basename(local_mask_path)}'
        
        # transfer the input files to the server
        server.exec_cmd(f'mkdir -p {remote_img_dir}/gum/')
        server.transfer_file_local2remote(local_rgb_img_path,remote_rgb_img_path)
        server.transfer_file_local2remote(local_mask_path,remote_mask_path)
        transfer_time = time.time()
        # print(f'[Transfer1 time]: {transfer_time - start_time} s')

        # gum
        remote_gum_script_dir = f'{remote_root_dir}/gum_package/'
        remote_gum_script_path = f'get_dxdyR.py'
        gum_cmd = f'cd {remote_gum_script_dir}; {remote_python_path} {remote_gum_script_path} -i {remote_rgb_img_path} -m {remote_mask_path} -model {self.model_path} -d {self.device}'
        server.exec_cmd(gum_cmd)
        exec_time = time.time()
        # print(f'exec time: {exec_time - transfer_time} s')

        # transfer the output dir to the server
        server.transfer_folder_remote2local(f'{remote_img_dir}/gum/', f'{os.path.dirname(local_rgb_img_path)}/')
        transfer_time = time.time()
        # print(f'[Transfer2 time]: {transfer_time - exec_time} s')

        # open gum_result.json to get dx,dy,R
        with open(f'{os.path.dirname(local_rgb_img_path)}/gum_result.json','r') as f:
            data = json.load(f)
            dx = data['dx']
            dy = data['dy']
            R = data['R']

        return dx,dy,R

if __name__ == "__main__":
    gum = GUM()
    image_path = r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/test/trajectory_000/1.png'
    mask_path = r'/media/datadisk10tb/leo/projects/realman-robot/open_door/data/test/trajectory_000/1/dtsam/center.png'
    dx, dy, R = gum.get_dxdyR(image_path,mask_path,root_dir='./gum_package/')
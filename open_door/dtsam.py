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

from utils.lib_io import *

class DTSAM():
    def __init__(self,classes='handle',device='cuda:0',threshold=0.3):
        self.classes = classes
        self.device = device
        self.threshold = threshold
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_dtsam.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.classes,cfg.device,cfg.threshold)

    def get_xy(self,img_path):
        from dtsam_package.detic_sam import detic_sam
        x,y,orientation = detic_sam(img_path,self.classes,self.device,self.threshold)
        return x,y,orientation

    def get_xy_server(self,img_path,server,remote_python_path,remote_root_dir,remote_img_dir):
        remote_dtsam_script_dir = f'{remote_root_dir}/dtsam_package/'
        remote_dtsam_script_path = f'detic_sam.py'
        local_img_path = img_path
        remote_img_path = f'{remote_img_dir}/{os.path.basename(local_img_path)}'

        # transfer the input image file to the server
        server.exec_cmd(f'mkdir -p {remote_img_dir}/dtsam/')
        server.transfer_file_local2remote(local_img_path,remote_img_path)

        # dtsam
        dtsam_cmd = f'cd {remote_dtsam_script_dir}; {remote_python_path} {remote_dtsam_script_path} -i {remote_img_path} -c {self.classes} -d {self.device} -t {self.threshold}'
        server.exec_cmd(dtsam_cmd)

        # transfer the output dir to the server
        server.transfer_folder_remote2local(f'{remote_img_dir}/dtsam/', f'{os.path.dirname(local_img_path)}/dtsam/')

        # open dtsam_result.json to get x and y
        with open(f'{os.path.dirname(local_img_path)}/dtsam/dtsam_result.json','r') as f:
            data = json.load(f)
            x = data['Cx']
            y = data['Cy']
            w = data['w']
            h = data['h']
            box = data['box']
            orientation = data['orientation']
        # print(f'Cx: {x}, Cy: {y}')
        # print(f'w: {w}, h: {h} orientation: {orientation}')
        # print(f'box:{box}')
        
        return x,y,orientation,w,h,box

    def process_images_server(self,img_path,server,remote_python_path,remote_root_dir,remote_img_dir):
        local_img_path = img_path
        
        remote_dtsam_script_dir = f'{remote_root_dir}/dtsam_package/'
        remote_dtsam_script_path = f'detic_sam.py'
        remote_img_path = f'{remote_img_dir}/{os.path.basename(local_img_path)}'

        # transfer the input image file to the server
        server.exec_cmd(f'mkdir -p {remote_img_dir}/dtsam/')
        server.transfer_file_local2remote(local_img_path,remote_img_path)

        # dtsam
        dtsam_cmd = f'cd {remote_dtsam_script_dir}; {remote_python_path} {remote_dtsam_script_path} -i {remote_img_path} -c {self.classes} -d {self.device} -t {self.threshold}'
        server.exec_cmd(dtsam_cmd)

        f_name = os.path.basename(local_img_path).split('.')[0]

        server.transfer_file_remote2local(f'{remote_img_dir}/dtsam/dtsam_result.json',f'{os.path.dirname(local_img_path)}/{f_name}.json',if_p=False)
        server.transfer_file_remote2local(f'{remote_img_dir}/dtsam/center.png',f'{os.path.dirname(local_img_path)}/{f_name}_mask.png',if_p=False)


if __name__ == "__main__":
    dtsam = DTSAM.init_from_yaml(cfg_path='cfg/cfg_dtsam.yaml')
import sys
root_dir = "../"
sys.path.append(root_dir)

import os
import time

from server import Server
from gum import GUM
from utils.lib_rgbd import *

## init
server = Server.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_server.yaml')
gum = GUM.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_gum.yaml')

## remote
remote_python_path = '/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python'
remote_root_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/'
remote_img_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/trajectory/remote/'

## image and mask
rgb_img_path = f'{root_dir}/example_img/ll_example_img3.png'
mask_path = f'{root_dir}/example_img/ll_example_img3_mask.png'
x1_2d,y1_2d = 985.1115095628415,437.5952868852459

rgb_img_path = f'{root_dir}/example_img/ll_example_img4.png'
mask_path = f'{root_dir}/example_img/ll_example_img4_mask.png'
x1_2d,y1_2d = 288.60552473846843,376.4644279810878

## crop img
start_time = time.time()
crop_rgb_img_path = f'{os.path.dirname(rgb_img_path)}/gum/rgb_cropped.png'
crop_image(rgb_img_path, center_x=x1_2d, center_y=y1_2d, new_w=gum.img_w, new_h=gum.img_h, save_path=crop_rgb_img_path)
crop_mask_path = f'{os.path.dirname(rgb_img_path)}/gum/mask_cropped.png'
mask_image = crop_image(mask_path, center_x=x1_2d, center_y=y1_2d, new_w=gum.img_w, new_h=gum.img_h, save_path=crop_mask_path)
crop_time = time.time()
print(f'[Crop Time] {crop_time - start_time} s')

## gum
dx,dy,R = gum.get_dxdyR_server(crop_rgb_img_path,crop_mask_path,server,remote_python_path,remote_root_dir,remote_img_dir)
print(f'[GUM Result] dx: {dx}, dy: {dy}, R: {R}')
gum_time = time.time()
print(f'[Hgum Time] {gum_time - crop_time} s')
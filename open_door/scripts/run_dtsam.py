import sys
root_dir = "../"
sys.path.append(root_dir)

import os
import time

from server import Server
from dtsam import DTSAM
from utils.lib_rgbd import *

## init
server = Server.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_server.yaml')
dtsam = DTSAM.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_dtsam.yaml')

## remote
remote_python_path = '/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python'
remote_root_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/'
remote_img_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/trajectory/remote/'

## image and mask
rgb_img_path = f'{root_dir}/example_img/ll_example_img1.png'

start_time = time.time()
x1_2d,y1_2d,orientation,w,h,box = dtsam.get_xy_server(rgb_img_path,server,remote_python_path,remote_root_dir,remote_img_dir)
print(f'[DTSAM Result] x1_2d: {x1_2d}, y1_2d: {y1_2d}, orientation: {orientation}, w: {w}, h: {h}, box: {box}')
dtsam_time = time.time()
print(f'[dtsam Time] {dtsam_time - start_time} s')
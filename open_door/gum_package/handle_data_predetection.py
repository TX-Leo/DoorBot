'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-15 23:36:56
Version: v1
File: 
Brief: 
'''
import os
import time

import sys
root_dir = "../"
sys.path.append(root_dir)
from utils.lib_io import *
from server import Server 
from dtsam import DTSAM

## os
# root_dir = r'./data/doorknob2/'
# names = get_filenames(folder=root_dir,is_base_name=False,filter='jpg')

root_dir = r'./data/crossbar2/'
names1 = get_filenames(folder=root_dir,is_base_name=False,filter='jpg')

root_dir = r'./data/drawer2/'
names2 = get_filenames(folder=root_dir,is_base_name=False,filter='jpg')

root_dir = r'./data/lever2/'
names3 = get_filenames(folder=root_dir,is_base_name=False,filter='jpg')

names = names1+names2+names3

## init
server = Server.init_from_yaml(cfg_path=f'../cfg/cfg_server.yaml')
dtsam = DTSAM.init_from_yaml(cfg_path=f'../cfg/cfg_dtsam.yaml')

## remote
remote_python_path = '/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python'
remote_root_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/'
remote_img_dir = '/media/datadisk10tb/leo/projects/realman-robot/open_door/trajectory/remote/'

## dtsam
start_time = time.time()
num = 0
for name in names:
    print(f'Process {os.path.basename(name)} ...')
    rgb_img_path = name
    dtsam.process_images_server(rgb_img_path,server,remote_python_path,remote_root_dir,remote_img_dir)
    num += 1
    print(f'[All Time] {time.time()-start_time} s')
    print(f'[Average Time] {(time.time()-start_time)/num} s')

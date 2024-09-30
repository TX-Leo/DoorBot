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
import re
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from handle_grasp_unlock_dataset import HandleGraspUnlockDataset
from handle_grasp_unlock_model import HandleGraspUnlockModel

import sys
root_dir = "../"
sys.path.append(root_dir)
from utils.lib_rgbd import *

RESNET_DEPTH = 18

def test(device):
    ## model
    model_load_path = r'./checkpoints/gum2.pth'
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True,device=device).to(device)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    root_dir = r'/media/datadisk10tb/leo/projects/data/drawer/test'

    image_files = sorted([f for f in os.listdir(root_dir) if re.match(r'.*_.*_.*_\d+\.png$', f)])
    
    for i in range(len(image_files)):
        print(f'[Num]: {i}')
        image_file = image_files[i]
        image_path = os.path.join(root_dir, image_file)
        mask_path = image_path.replace('.png', '_mask.png')

        ## forward
        dx,dy,R = model.gum_api(image_path,mask_path,if_p=False)

        ## vis_grasp
        with open(image_path.replace('.png','.json'), 'r') as f:
            data = json.load(f)
        Cx = data['Cx']
        Cy = data['Cy']
        orientation = data['orientation']
        x1_2d, y1_2d = Cx+dx, Cy+dy
        angle = 90
        x2_2d, y2_2d, Ox, Oy = rotate_point(x1_2d, y1_2d, R, orientation, angle)
        vis_grasp(image_path, dx, dy, x1_2d, y1_2d, x2_2d, y2_2d, Ox, Oy, R, orientation, angle, save_path=image_path.replace('.png','_vis_predicted.png'))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test(device)
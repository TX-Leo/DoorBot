'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-19 13:42:57
Version: v1
File: 
Brief: 
'''
import os
import json
import argparse
from PIL import Image
import torch
import time

import sys
root_dir = "./gum_package"
sys.path.append(root_dir)
    
from handle_grasp_unlock_model import HandleGraspUnlockModel

RESNET_DEPTH = 18

def get_dxdyR(image_path='',mask_path='',model_path='checkpoints/gum8.pth',device='cuda:0',root_dir='./',if_p=False):
    start_time = time.time()
    ## model
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True).to(device)
    model_path = f'{root_dir}/{model_path}'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    load_time = time.time()
    # print(f'[load_time]: {load_time-start_time} s')

    ## image dir
    image_dir = os.path.dirname(image_path)+'/gum'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    ## mask path
    if not mask_path:
        mask_path = image_path.replace('.png', '_mask.png')
    os_time = time.time()
    # print(f'[os_time]: {os_time-load_time} s')

    ## forward
    dx,dy,R = model.gum_api(image_path,mask_path,if_p=if_p)
    api_time = time.time()
    # print(f'[api_time]: {api_time-os_time} s')

    ## save to gum/gum_result.json
    result_save_path = image_dir+'/gum_result.json'
    result = {"dx":dx,
              "dy":dy,
              "R":R,
    }
    with open(result_save_path, 'w') as file:
        json.dump(result, file, indent=4)
    json_time = time.time()
    # print(f'[json_time]: {json_time-api_time} s')

    return dx,dy,R

def main(args):
    get_dxdyR(args.image_path,args.mask_path,args.model_path,args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, default="", help="Input image path.")
    parser.add_argument("-m", "--mask_path", type=str, default="", help="Input mask path.")
    parser.add_argument("-model", "--model_path", type=str, default="./checkpoints/gum.pth", help="Input model path.")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Input device.")
    main(parser.parse_args())
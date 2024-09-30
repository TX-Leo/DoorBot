'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-18 13:02:26
Version: v1
File: 
Brief: 
'''
import clip
from PIL import Image
import torch

import sys
root_dir = '../'
sys.path.append(root_dir)

from utils.lib_io import *

class CLIP(object):
    def __init__(self, model_name = "ViT-B/32", device = 'cuda:0'):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.create_model()

    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_clip.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.model_name,cfg.device)

    def create_model(self):
        self.clip_model, self.preprocess = clip.load(self.model_name, device=self.device) # Load CLIP model

    def get_all_models(self,if_p=False):
        available_models = clip.available_models()
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        if if_p:
            print(available_models)

    def clip_api(self,rgb_img, text_prompt, if_p = False):
        if isinstance(rgb_img,str):
            rgb_img_input = self.preprocess(Image.open(rgb_img)).unsqueeze(0).to(self.device)# get rgb_img_input
        else:
            rgb_img_input = self.preprocess(rgb_img).unsqueeze(0).to(self.device)# get rgb_img_input
        text_input = clip.tokenize(text_prompt).to(self.device)# get text_input
        with torch.no_grad():
            image_features = self.clip_model.encode_image(rgb_img_input)
            text_features = self.clip_model.encode_text(text_input)
            logits_per_image, logits_per_text = self.clip_model(rgb_img_input, text_input)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            if if_p:
                print("Label probs:", probs)  # prints: [[0.9927937  0.00421068]]
        return probs

    def clip_detection(self,rgb_img=None,text_prompt=["door that is closed", "door that is open"],if_p=False):
        probs = self.clip_api(rgb_img, text_prompt) # using CLIP
        result = 1 if probs[0][1] > probs[0][0] else 0 # Reward is 1 if closer to 'open door' prompt, 0 otherwise
        result_text = text_prompt[result]
        result_probs = probs[0]
        if if_p:
            # print("result:", result)  # prints: 1/0
            print(f'[CLIP INFO] Result: {result_text}\tLabel probs: {result_probs}') # door is open/closed
        return result,result_text,result_probs

if __name__ == '__main__':
    _clip = CLIP.init_from_yaml(cfg_path='../cfg/cfg_clip.yaml')
    rgb_img = '../test_img/1.png'
    _clip.clip_detection(rgb_img,text_prompt=["door that is closed", "door that is open"],if_p=True)
import time
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

RESNET_DEPTH = 18

import sys
root_dir = "../"
sys.path.append(root_dir)
from utils.lib_io import *

class HandleGraspUnlockModel(nn.Module):
    def __init__(self, resnet_depth=18, pretrained=True,device='cuda:0'):
        super(HandleGraspUnlockModel, self).__init__()

        ## reset
        self.resnet = models.__dict__[f'resnet{resnet_depth}'](pretrained=pretrained)
        
        self.image_encoder = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last FC layer
        self.mask_encoder = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last FC layer

        self.feature_dim = self.resnet.fc.in_features

        self.predictor = nn.Sequential( 
            nn.Linear(2 * self.feature_dim, 512),  # merge tow resnet features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # output: dx, dy, R
        )

        self.device = device

    def forward(self, image, mask):
        image_features = self.image_encoder(image.to(self.device)).squeeze() # batch_size * 512
        mask_features = self.mask_encoder(mask.to(self.device)).squeeze() # batch_size * 512
        
        # check dimension
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0).flatten(start_dim=1)  # batch_size * 512
        if len(mask_features.shape) == 1:
            mask_features = mask_features.unsqueeze(0).flatten(start_dim=1)  # batch_size * 512

        features = torch.cat((image_features, mask_features), dim=1)
        output = self.predictor(features)

        return output
    
    def gum_api(self,image_path,mask_path,if_p=False):
        start_time = time.time()
        self.eval()
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        ## transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image) # 3 * 224 * 224
        mask = transform(mask) # 3 * 224 * 224

        ## add batch dimension
        image = image.unsqueeze(0) # 1 * 3 * 224 * 224
        mask = mask.unsqueeze(0) # 1 * 3 * 224 * 224
        image_transform_time = time.time()
        # print(f'[image_transform_time]: {image_transform_time-start_time} s')

        ## forward
        with torch.no_grad():
            output = self.forward(image,mask)
        dx, dy, R = output[0].cpu().numpy()
        dx = float(dx)
        dy = float(dy)
        R = float(R)
        forward_time = time.time()
        # print(f'[forward_time]: {forward_time-image_transform_time} s')

        if if_p:
            print(f'[GUM Result] dx: {dx}, dy: {dy}, R: {R}')
        
        return dx,dy,R
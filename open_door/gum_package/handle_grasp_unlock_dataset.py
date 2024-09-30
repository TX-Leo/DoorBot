'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-19 13:27:25
Version: v1
File: 
Brief: 
'''
import os
import re
import json
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HandleGraspUnlockDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(self.root_dir) if re.match(r'.*_.*_.*_\d+\.png$', f)])
        # self.image_files = sorted([f for f in os.listdir(self.root_dir) if re.match(r'.*_.*_\d+\.png$', f)])
        self.transform = transform
        self.num = len(self.image_files)

    def __len__(self):
        return self.num

    def __str__(self):
        print(f'Num of dataset is {self.num}')
        return ''

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        
        # Load annotation data
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(self.root_dir, json_file)

        # Skip if JSON file does not exist
        if not os.path.exists(json_path):
            return None

        # load original rgb image and mask image
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(image_path.replace('.jpg','_mask.png')).convert("RGB")

        # transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image = self.transform(image)
        mask = self.transform(mask)
        
        # Load annotations from JSON file
        with open(json_path, 'r') as f:
            annotations = json.load(f)
            if 'dx' not in annotations: # not annotated
                return None
        
        dx = annotations["dx"]
        dy = annotations["dy"]
        R = annotations["R"]
        
        target = torch.tensor([dx, dy, R], dtype=torch.float32)

        return image, mask, target

if __name__ == "__main__":
    dataset_dir = r'./data/drawer/original'
    dataset = HandleGraspUnlockDataset(root_dir=dataset_dir)
    print(dataset)
    dataset = [data for data in dataset if data is not None]
    print(len(dataset))

    # lever:    968->684
    # crossbar: 204->171
    # doorknob: 471->331
    # drawer:   159->117
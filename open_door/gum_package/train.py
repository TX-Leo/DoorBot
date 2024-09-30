'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-19 13:34:10
Version: v1
File: 
Brief: 
'''
import time
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from handle_grasp_unlock_dataset import HandleGraspUnlockDataset
from handle_grasp_unlock_model import HandleGraspUnlockModel

BATCH_SIZE = 16
RESNET_DEPTH = 18
LR = 1e-4
N_EPOCHS = 50

def train(device):
    ## dataset and dataloader
    # train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/lever/train'
    # train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    # print(train_dataset)
    # train_dataset1 = [data for data in train_dataset if data is not None]

    # train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/doorknob/train'
    # train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    # print(train_dataset)
    # train_dataset2 = [data for data in train_dataset if data is not None]
    
    # train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/drawer/train'
    # train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    # print(train_dataset)
    # train_dataset3 = [data for data in train_dataset if data is not None]
    
    train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/crossbar/train'
    train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    print(train_dataset)
    train_dataset4 = [data for data in train_dataset if data is not None]
    
    # train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/lever/test'
    # train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    # print(train_dataset)
    # train_dataset5 = [data for data in train_dataset if data is not None]

    # train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/doorknob/test'
    # train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    # print(train_dataset)
    # train_dataset6 = [data for data in train_dataset if data is not None]
    
    # train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/drawer/test'
    # train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    # print(train_dataset)
    # train_dataset7 = [data for data in train_dataset if data is not None]
    
    train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/crossbar/test'
    train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    print(train_dataset)
    train_dataset8 = [data for data in train_dataset if data is not None]

    # train_dataset = train_dataset1+train_dataset2+train_dataset3+train_dataset4+train_dataset5+train_dataset6+train_dataset7+train_dataset8
    
    # train_dataset_dir = r'/media/datadisk10tb/leo/projects/data/lever/train_small'
    # train_dataset = HandleGraspUnlockDataset(root_dir=train_dataset_dir)
    # print(train_dataset)
    # train_dataset = [data for data in train_dataset if data is not None]


    train_dataset = train_dataset4+train_dataset8

    model_load_path = r'./checkpoints/gum18.pth'
    loss_save_path = r'./checkpoints/loss18.png'

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    ## model
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True,device=device).to(device)
    model.train()

    ## loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    plt.ion()
    fig, ax = plt.subplots()

    start_time = time.time()

    for epoch in range(N_EPOCHS):
        for i, (images, masks, targets) in enumerate(train_dataloader):
            ## forward
            outputs = model(images, masks)
            
            ## backward
            loss = loss_fn(outputs, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ## print
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{N_EPOCHS}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                ## vis
                losses.append(loss.item())
                ax.clear()
                ax.plot(losses)
                ax.set_title('Training Loss')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                plt.pause(0.1)
        plt.ioff()
        fig.savefig(loss_save_path)

        # ## save model
        torch.save(model.state_dict(), model_load_path)
        print(f'[All Time] {time.time()-start_time}')
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(device)
'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-09-18 23:24:23
Version: v1
File: 
Brief: 
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from handle_grasp_unlock_dataset import HandleGraspUnlockDataset
from handle_grasp_unlock_model import HandleGraspUnlockModel

BATCH_SIZE = 16
RESNET_DEPTH = 18

def eval():
    eval_dataset_dir = r'/media/datadisk10tb/leo/projects/data/lver/test'
    model_load_path = r'./checkpoints/gum8.pth'
    eval_vis_path = r'./checkpoints/'

    ## dataset and dataloader
    eval_dataset = HandleGraspUnlockDataset(root_dir=eval_dataset_dir)
    print(eval_dataset)
    eval_dataset = [data for data in eval_dataset if data is not None]
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ## model
    model = HandleGraspUnlockModel(resnet_depth=RESNET_DEPTH, pretrained=True,device=device).to(device)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    
    total_loss = 0.0
    loss_fn = nn.MSELoss()

    relative_errors = []
    all_real_values = []
    all_predicted_values = []

    with torch.no_grad():
        for i, (images, masks, targets) in enumerate(eval_dataloader):
            outputs = model(images, masks)

            loss = loss_fn(outputs, targets.to(device))
            total_loss += loss.item()

            all_real_values.extend(targets.cpu().numpy())
            all_predicted_values.extend(outputs.cpu().detach().numpy())

            for j in range(targets.size(0)):
                real_dx, real_dy, real_R = targets[j].cpu().numpy()

                ## forward
                predicted_dx, predicted_dy, predicted_R = outputs[j].cpu().detach().numpy()

                ## error
                if real_dx != 0:
                    relative_error_dx = abs((predicted_dx - real_dx) / real_dx)
                else:
                    # relative_error_dx = float('inf')
                    relative_error_dx = 0
                if real_dy != 0:
                    relative_error_dy = abs((predicted_dy - real_dy) / real_dy)
                else:
                    # relative_error_dy = float('inf')
                    relative_error_dy = 0
                if real_R != 0:
                    relative_error_R = abs((predicted_R - real_R) / real_R)
                else:
                    # relative_error_R = float('inf')
                    relative_error_R = 0
                print(f'relative_error_dx: {relative_error_dx}')
                print(f'relative_error_dy: {relative_error_dy}')
                print(f'relative_error_R: {relative_error_R}')
                relative_errors.append([relative_error_dx, relative_error_dy, relative_error_R])

            print(f'Batch [{i+1}/{len(eval_dataloader)}], Loss: {loss.item():.4f}')

    average_loss = total_loss / len(eval_dataloader)
    print(f'Average eval Loss: {average_loss:.4f}')

    # Calculate the average relative error for all samples
    relative_errors = np.array(relative_errors)
    mean_relative_errors = np.mean(relative_errors, axis=0)
    print(f"Mean Relative Errors (dx, dy, R): {mean_relative_errors}")

    labels = ['dx', 'dy', 'R']
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(15, 5))

    for i in range(3):
        plt.figure()
        
        real_values = [val[i] for val in all_real_values]
        predicted_values = [val[i] for val in all_predicted_values]

        x = np.arange(len(real_values))

        plt.bar(x - 0.2, real_values, width=0.4, label='Real', color='red')
        plt.bar(x + 0.2, predicted_values, width=0.4, label='Predicted', color='green')

        plt.xlabel('Data Index')
        plt.ylabel(f'{labels[i]} Value')
        plt.title(f'Real vs. Predicted {labels[i]}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(eval_vis_path, f"eval_vis_{labels[i]}_lver_8.png"))
        plt.close()
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval(device)
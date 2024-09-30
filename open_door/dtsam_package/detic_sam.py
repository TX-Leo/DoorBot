# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import json
import os
import pickle
import random
from typing import List, Dict, Tuple
import sys
import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d
from PIL import Image, ImageDraw
import torch
import time
# Change the current working directory to 'Detic'

try:
    os.chdir('Detic')
except:
    current_directory = os.getcwd()
    root_dir = f'{current_directory}/open_door/dtsam_package'
    os.chdir(f'{root_dir}/Detic')

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
sys.path.insert(0, 'third_party/CenterNet2/')

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries 
from Detic.detic.modeling.text.text_encoder import build_text_encoder
from collections import defaultdict
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
# SAM libraries
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

def DETIC_predictor():
    # Build the detector and download our pretrained weights
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1 # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    cfg.MODEL.DEVICE='cpu' # 'cuda' or 'cpu'
    detic_predictor = DefaultPredictor(cfg)
    return detic_predictor

def default_vocab():
    detic_predictor = DETIC_predictor()
    # Setup the model's vocabulary using build-in datasets
    BUILDIN_CLASSIFIER = {
        'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
        'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
        'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
        'coco': 'datasets/metadata/coco_clip_a+cname.npy',
    }

    BUILDIN_METADATA_PATH = {
        'lvis': 'lvis_v1_val',
        'objects365': 'objects365_v2_val',
        'openimages': 'oid_val_expanded',
        'coco': 'coco_2017_val',
    }

    vocabulary = 'openimages' # change to 'lvis', 'objects365', 'openimages', or 'coco'
    metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
    classifier = BUILDIN_CLASSIFIER[vocabulary]
    num_classes = len(metadata.thing_classes)
    reset_cls_test(detic_predictor.model, classifier, num_classes)

def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def visualize_detic(output):
    output_im = output.get_image()[:, :, ::-1]
    cv2.imshow("Detic Predictions", output_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def custom_vocab(detic_predictor, classes, threshold=0.3):
    vocabulary = 'custom'
    metadata = MetadataCatalog.get("__unused2")
    metadata.thing_classes = classes # Change here to try your own vocabularies!
    classifier = get_clip_embeddings(metadata.thing_classes)
    num_classes = len(metadata.thing_classes)
    reset_cls_test(detic_predictor.model, classifier, num_classes)

    # Reset visualization threshold
    output_score_threshold = threshold
    for cascade_stages in range(len(detic_predictor.model.roi_heads.box_predictor)):
        detic_predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
    return metadata

def Detic(im, metadata, detic_predictor, visualize=False):
    if im is None:
        print("Error: Unable to read the image file")

    # Run model and show results
    output =detic_predictor(im[:, :, ::-1])  # Detic expects BGR images.
    v = Visualizer(im, metadata)
    out = v.draw_instance_predictions(output["instances"].to('cpu'))
    instances = output["instances"].to('cpu')
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    if visualize:
        visualize_detic(out)
    return boxes, classes

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def visualize_output(im, masks, input_boxes, classes, image_save_path, mask_only=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    if not mask_only:
        for box, class_name in zip(input_boxes, classes):
            show_box(box, plt.gca())
            x, y = box[:2]
            plt.gca().text(x, y - 5, class_name, color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='green', edgecolor='green', alpha=0.5))
    plt.axis('off')
    plt.savefig(image_save_path)
    #plt.show()

def SAM_predictor(device):
    sam_checkpoint = "../segment-anything/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def SAM(im, boxes, class_idx, metadata, sam_predictor):
    sam_predictor.set_image(im)
    input_boxes = torch.tensor(boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, im.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks

def generate_colors(num_colors):
    hsv_colors = []
    for i in range(num_colors):
        hue = i / float(num_colors)
        hsv_colors.append((hue, 1.0, 1.0))

    return [mcolors.hsv_to_rgb(color) for color in hsv_colors]

def process_masks(masks,mask_image_save_path,center_image_save_path):
    mask = masks[0].cpu().numpy()[0]
    save_mask(mask,mask_image_save_path)
    Cx,Cy = get_mask_center(mask,save_path=None)
    add_point_to_image(image_path=mask_image_save_path, x=Cx, y=Cy, dot_size=5, dot_color=(255,0,0),save_path=center_image_save_path)
    return Cx,Cy

def save_mask(mask,mask_image_save_path):
    h, w = mask.shape[-2:]
    mask_image = np.where(mask.reshape(h, w) > 0, 255, 0).astype(np.uint8)
    imageio.imwrite(mask_image_save_path, mask_image)
    return mask_image

def get_mask_center(mask,save_path=None):
    # Convert mask array to binary image
    binary_image = mask.astype('uint8')  # Converts a Boolean array to an unsigned 8-bit integer type

    # Calculate moments
    M = cv2.moments(binary_image)

    # Calculate centroid coordinates
    if M['m00'] != 0:
        # Calculate centroid coordinates
        Cx = M['m10'] / M['m00']
        Cy = M['m01'] / M['m00']
    else:
        # Set centroid coordinates to invalid values
        Cx = -1
        Cy = -1
    
    # save
    if save_path:
        data = {'Cx': Cx,
                'Cx': Cy}
        with open(save_path, 'w') as file:
            json.dump(data, file, indent=4)
    return Cx, Cy

def add_point_to_image(image_path, x, y, dot_size=1, dot_color=(255, 0, 0),save_path=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Calculate the coordinates for the bounding box of the dot
    x1 = x - dot_size
    y1 = y - dot_size
    x2 = x + dot_size
    y2 = y + dot_size
    # Draw the dot on the image
    draw.ellipse((x1, y1, x2, y2), fill=dot_color)
    # Save the modified image
    if save_path:
        image.save(save_path)

def mask_to_image(mask, save_path=None):
    from PIL import Image
    # Convert mask tensor to numpy array
    mask_np = mask.detach().cpu().numpy()
    # Convert mask array to binary image
    binary_image = (mask_np * 255).astype('uint8')
    # Create PIL Image from the binary image
    image = Image.fromarray(binary_image, mode='L')
    # Save the image to a JPEG file
    if save_path:
        image.save(save_path)
    return image

def test_process_2d_mask(mask):
    # Convert mask to image
    image = mask_to_image(mask, save_path='images/mask_image.jpg')
    # Get the center point of the mask
    Cx, Cy = get_mask_center(mask)
    # Print the center coordinates
    print("Center:", Cx, Cy)

def detic_sam(image_path,classes='handle',device='cuda:0',threshold=0.3):
    start_time = time.time()

    # We are one directory up in Detic.
    image_path = os.path.join("..", image_path)
    image_dir = os.path.dirname(image_path)+'/dtsam'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # print(f'image_path:{image_path}')
    # print(f'image_dir:{image_dir}')

    # open image
    image = Image.open(image_path)
    image = np.array(image, dtype=np.uint8)
    
    # crete predictor
    detic_predictor = DETIC_predictor()
    sam_predictor = SAM_predictor(device)
    metadata = custom_vocab(detic_predictor, classes,threshold)
    predictor_time = time.time() - start_time
    # print(f'[predictor_time]: {predictor_time} s')

    # detect
    boxes, class_idx = Detic(image, metadata, detic_predictor)
    detic_detect_time = time.time() - predictor_time
    # print(f'[detic_detect_time]: {detic_detect_time} s')

    # assert len(boxes) > 0, "Zero detections."

    result_save_path = image_dir+'/dtsam_result.json'

    if len(boxes) == 0:
        print(f'Zero detections.')
        w = 0
        h = 0
        Cx = 0
        Cy = 0
        orientation = ''
        box = [0,0,0,0]
    else:
        masks = SAM(image, boxes, class_idx, metadata, sam_predictor)
        sam_detect_time = time.time() - detic_detect_time
        # print(f'[sam_detect_time]: {sam_detect_time} s')

        # Save detections as a png. Save only segmentation without bounding box as a separate image.
        classes = [metadata.thing_classes[idx] for idx in class_idx]
        bbox_image_save_path = image_dir+'/bbox.png'
        segm_image_save_path = image_dir+'/segm.png'
        mask_image_save_path = image_dir+'/mask.png'
        center_image_save_path = image_dir+'/center.png'
        visualize_output(image, masks, boxes, classes, bbox_image_save_path)
        visualize_output(image, masks, boxes, classes, segm_image_save_path, mask_only=True)
        Cx, Cy = process_masks(masks,mask_image_save_path,center_image_save_path)
        orientation = 'horizontal'
        box = [float(value) for value in boxes[0].tolist()]
        w, h = box[2] - box[0], box[3] - box[1]
        orientation = 'horizontal' if w >= h else 'vertical'
        print(f'Cx: {Cx}, Cy: {Cy}')
        print(f'w: {w}, h: {h} orientation: {orientation}')
        print(f'box:{box}')
    
    result = {"w":w,
              "h":h,
              "box":box,
              "Cx":Cx,
              "Cy":Cy,
              "orientation":orientation
    }
    with open(result_save_path, 'w') as file:
        json.dump(result, file, indent=4)
    
    return Cx,Cy,orientation

def main(args):
    detic_sam(args.image_path,args.classes,args.device,args.threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, default="/media/datadisk10tb/leo/projects/realman-robot/open_door/images/image1/rgb.png", help="Input image path.")
    parser.add_argument("-c", "--classes", nargs="+", default='handle', help="List of classes to detect. Each class can be a word or a sentence.")
    parser.add_argument("-d", "--device", type=str, default="cuda:1", help="Device to run on.")
    parser.add_argument("-t", "--threshold", type=float, default=0.3, help="detection score threshold")
    main(parser.parse_args())

import os
import json
import re
from PIL import Image
import random
import numpy as np

import sys
root_dir = "../"
sys.path.append(root_dir)
from utils.lib_rgbd import *

class HandleDataAugmentator:
    def __init__(self, input_dir, output_train_dir, output_test_dir, train_ratio=0.95, translation_x_min=-30, translation_x_max=30,
                 translation_y_min=-30, translation_y_max=30, translation_num=5,
                 ratio_min=0.8, ratio_max=1.2, resize_num=5):
        """
        Initializes the DataAugmentator class with augmentation parameters.

        Args:
            input_dir (str): The directory containing the input images and JSON files.
            output_train_dir (str): The directory to save the augmented data.
            output_test_dir (str): The directory to save the augmented data.
            translation_x_min (int, optional): Minimum translation in x-axis. Defaults to -50.
            translation_x_max (int, optional): Maximum translation in x-axis. Defaults to 50.
            translation_y_min (int, optional): Minimum translation in y-axis. Defaults to -50.
            translation_y_max (int, optional): Maximum translation in y-axis. Defaults to 50.
            translation_num (int, optional): Number of random translations to perform. Defaults to 5.
            ratio_min (float, optional): Minimum resize ratio. Defaults to 0.8.
            ratio_max (float, optional): Maximum resize ratio. Defaults to 1.2.
            resize_num (int, optional): Number of random resizes to perform. Defaults to 5.
        """
        self.input_dir = input_dir
        self.output_train_dir = output_train_dir
        self.output_test_dir = output_test_dir
        self.train_ratio = train_ratio
        self.translation_x_min = translation_x_min
        self.translation_x_max = translation_x_max
        self.translation_y_min = translation_y_min
        self.translation_y_max = translation_y_max
        self.translation_num = translation_num
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.resize_num = resize_num
        self.crop_width = 640
        self.crop_height = 640
        self.sum = 0

        if not os.path.exists(self.output_train_dir):
            os.makedirs(self.output_train_dir)
        if not os.path.exists(self.output_test_dir):
            os.makedirs(self.output_test_dir)

    def augment_data(self):
        """
        Augments the images and annotations in the input directory.
        """
        image_path_lst = []
        mask_path_lst = []
        data_lst = []

        for filename in os.listdir(self.input_dir):
            if filename.endswith(".jpg") and re.match(r'^\d+\.jpg$', filename):
                image_path = os.path.join(self.input_dir, filename)
                mask_path = os.path.join(self.input_dir, os.path.splitext(filename)[0] + "_mask.png")
                json_path = os.path.join(self.input_dir, os.path.splitext(filename)[0] + ".json")

                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    if 'dx' not in data: # not annotated
                        continue
                    
                    image_path_lst.append(image_path)
                    mask_path_lst.append(mask_path)
                    data_lst.append(data)
                    self.sum += 1

        print(f'[Sum]: {self.sum}')
        for i in range(self.sum):
            print(f'[num_{i}] ...')
            if i <= int(self.train_ratio*self.sum):
                output_dir = self.output_train_dir
            else: 
                output_dir = self.output_test_dir
            self.augment_single_image(image_path_lst[i],mask_path_lst[i],data_lst[i],output_dir)

    def visualization(self,data,img_path,save_path):
        if data['R'] != 0:
            x1_2d, y1_2d = data['Cx'] + data['dx'], data['Cy'] + data['dy']
            dx = data['dx']
            dy = data['dy']
            R = data['R']
            orientation = data['orientation']
            angle = 90
            x2_2d, y2_2d, Ox, Oy = rotate_point(x1_2d, y1_2d, R, orientation, angle)
            vis_grasp(img_path, dx, dy, x1_2d, y1_2d, x2_2d, y2_2d, Ox, Oy, R, orientation, angle, save_path)

    def augment_single_image(self,image_path,mask_path,data,output_dir):
        """
        Augments a single image and its annotations.

        Args:
            image_path (str): Path to the image file.
            data (dict): Dictionary containing the image annotations.
        """
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        original_width, original_height = image.size

        for i in range(self.translation_num):
            for j in range(self.resize_num):
                for k in range(2):
                    # 1. Flip image and annotations
                    if k == 0:
                        flipped_image = image
                        flipped_mask = mask
                        flipped_data = data
                    else:
                        if data['orientation'] == 'horizontal':
                            flip_mode = Image.FLIP_LEFT_RIGHT
                        elif data['orientation'] == 'vertical':
                            flip_mode = Image.FLIP_TOP_BOTTOM
                        flipped_image = image.transpose(flip_mode)
                        flipped_mask = mask.transpose(flip_mode)
                        flipped_data = self.adjust_annotations_for_flipping(data.copy(),original_width,original_height)
                    
                    # # Save flipped image and JSON
                    # new_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}_flipped.png"
                    # new_filepath = os.path.join(output_dir, new_filename)
                    # flipped_image.save(new_filepath)
                    # new_json_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}_flipped.json"
                    # with open(os.path.join(output_dir, new_json_filename), 'w') as f:
                    #     json.dump(flipped_data, f, indent=4)
                    # self.visualization(flipped_data,new_filepath,new_filepath.replace('.png', '_vis.png'))
                    
                    # 2. Resize image and annotations
                    
                    # Random resize
                    ratio = random.uniform(self.ratio_min, self.ratio_max)

                    new_width = int(original_width * ratio)
                    new_height = int(original_height * ratio)
                    resized_image = flipped_image.resize((new_width, new_height))
                    resized_mask = flipped_mask.resize((new_width, new_height))
                    resized_data = self.adjust_annotations(flipped_data.copy(), 0, 0, ratio)

                    # # Save resized image and JSON
                    # new_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}_resized.png"
                    # new_filepath = os.path.join(output_dir, new_filename)
                    # resized_image.save(new_filepath)
                    # new_json_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}_resized.json"
                    # with open(os.path.join(output_dir, new_json_filename), 'w') as f:
                    #     json.dump(resized_data, f, indent=4)
                    # self.visualization(resized_data,new_filepath,new_filepath.replace('.png', '_vis.png'))

                    # 3. Translate image and annotations

                    # Random translation
                    tx = random.randint(self.translation_x_min, self.translation_x_max)
                    ty = random.randint(self.translation_y_min, self.translation_y_max)

                    translated_data = self.adjust_annotations(resized_data.copy(), tx, ty, 1)
                    translated_image = Image.new("RGB", (new_width, new_height))
                    translated_mask = Image.new("RGB", (new_width, new_height))
                    translated_image.paste(resized_image, (tx, ty))
                    translated_mask.paste(resized_mask, (tx, ty))

                    # # Save translated image and JSON
                    # new_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}_translated.png"
                    # new_filepath = os.path.join(output_dir, new_filename)
                    # translated_image.save(new_filepath)
                    # new_json_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}_translated.json"
                    # with open(os.path.join(output_dir, new_json_filename), 'w') as f:
                    #     json.dump(translated_data, f, indent=4)
                    # self.visualization(translated_data,new_filepath,new_filepath.replace('.png', '_vis.png'))

                    # 4. Crop image and annotations
                    crop_x_min = translated_data['Cx'] - self.crop_width // 2
                    crop_y_min = translated_data['Cy'] - self.crop_height // 2
                    crop_x_max = crop_x_min + self.crop_width
                    crop_y_max = crop_y_min + self.crop_height

                    # Adjust crop coordinates to stay within image boundaries
                    if crop_x_min < 0:
                        crop_x_min = 0
                        crop_x_max = self.crop_width
                    elif crop_x_max > new_width:
                        crop_x_max = new_width
                        crop_x_min = new_width - self.crop_width

                    if crop_y_min < 0:
                        crop_y_min = 0
                        crop_y_max = self.crop_height
                    elif crop_y_max > new_height:
                        crop_y_max = new_height
                        crop_y_min = new_height - self.crop_height

                    # Crop image
                    cropped_image = translated_image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                    cropped_mask = translated_mask.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

                    # Adjust annotations for cropping
                    cropped_data = self.adjust_annotations_for_cropping(translated_data.copy(), crop_x_min, crop_y_min)

                    # Save cropped image(original)
                    new_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}.png"
                    new_filepath = os.path.join(output_dir, new_filename)
                    cropped_image.save(new_filepath)

                    # Save cropped image(mask)
                    new_mask_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}_mask.png"
                    cropped_mask.save(os.path.join(output_dir, new_mask_filename))

                    # Save JSON
                    new_json_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_{k}.json"
                    with open(os.path.join(output_dir, new_json_filename), 'w') as f:
                        json.dump(cropped_data, f, indent=4)
                    
                    # Save vis image
                    self.visualization(cropped_data,new_filepath,new_filepath.replace('.png', '_vis.png'))


    def adjust_annotations(self, data, tx, ty, ratio):
        """
        Adjusts the annotations based on translation and resize.

        Args:
            data (dict): Dictionary containing the image annotations.
            tx (int): Translation in x-axis.
            ty (int): Translation in y-axis.
            ratio (float): Resize ratio.

        Returns:
            dict: Dictionary containing the adjusted annotations.
        """
        data['box'][0] = (data['box'][0] + tx) * ratio
        data['box'][1] = (data['box'][1] + ty) * ratio
        data['box'][2] = (data['box'][2] + tx) * ratio
        data['box'][3] = (data['box'][3] + ty) * ratio

        data['Cx'] = (data['Cx'] + tx) * ratio
        data['Cy'] = (data['Cy'] + ty) * ratio

        data['dx'] = (data['dx']) * ratio
        data['dy'] = (data['dy']) * ratio
        data['R'] = (data['R'])* ratio

        data['w'] = (data['w']) * ratio
        data['h'] = (data['h']) * ratio

        return data
    
    def adjust_annotations_for_flipping(self,data,img_w,img_h):
        if data['orientation'] == 'horizontal':
            data['box'][0] = img_w - data['box'][0]
            data['box'][1] = data['box'][1]
            data['box'][2] = img_w - data['box'][2]
            data['box'][3] = data['box'][3]
            data['Cx'] = img_w - data['Cx']
            data['Cy'] = data['Cy']
            data['dx'] = -data['dx']
            data['dy'] = data['dy']
            data['R'] = -data['R']

        elif data['orientation'] == 'vertical':
            data['box'][0] = data['box'][0]
            data['box'][1] = img_h - data['box'][1]
            data['box'][2] = data['box'][2]
            data['box'][3] = img_h - data['box'][3]
            data['Cx'] = data['Cx']
            data['Cy'] = img_h - data['Cy']
            data['dx'] = data['dx']
            data['dy'] = -data['dy']
            data['R'] = -data['R']
    
        # w,h stay the same
        return data

    def adjust_annotations_for_cropping(self, data, crop_x_min, crop_y_min):
        """
        Adjusts the annotations for cropping.

        Args:
            data (dict): Dictionary containing the image annotations.
            crop_x_min (int): Starting x-coordinate of the cropping region.
            crop_y_min (int): Starting y-coordinate of the cropping region.

        Returns:
            dict: Dictionary containing the adjusted annotations.
        """
        data['box'][0] -= crop_x_min
        data['box'][1] -= crop_y_min
        data['box'][2] -= crop_x_min
        data['box'][3] -= crop_y_min

        data['Cx'] -= crop_x_min
        data['Cy'] -= crop_y_min
        
        # w,h,dx,dy,R stay the same
        
        return data

if __name__ == "__main__":
    root_dir = './data/lever'
    input_dir = f'{root_dir}/original/'
    output_train_dir = f'{root_dir}/train'
    output_test_dir = f'{root_dir}/test/'
    augmentator = HandleDataAugmentator(input_dir, output_train_dir, output_test_dir)
    augmentator.augment_data()
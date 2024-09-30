<!--
 * @Author: TX-Leo
 * @Mail: tx.leo.wz@gmail.com
 * @Date: 2024-09-20 13:46:14
 * @Version: v1
 * @File: 
 * @Brief: 
-->
GUM (Grasping-and-Unlocking Model)
=======================================

## Description
We propose GUM which can be used for robotic opening-door task. 
We collected 1303 rgb images from the Internet and Real-World, and annotated them manually. Then we conducted dataset augmentation to enlarge the size of original dataset.
Finally, we trained this model on the augmented dataset.

This model demonstrated that it has a great ability to have a good performance in grasping and unlocking part of the whole task.

We also proved that even a small 2d dataset purely from internet can train a good model for the complicated 3d robotic manipulation task in the uncertain open-world environments.

The model tasks the raw rgb image and the mask image of object which can be gotten by existing methods such as Detic and SAM as the input. Then it will generate the corresponding grasping and unlocking parameters. Details can be checked in the paper.

## Usage
- Dataset: 
  - `internet_data_url.txt` records all the source of the internet data.
  - `handle_data_preprocess.py` preprocesses all raw rgb images.
  - `handle_data_predetection.py` processes all raw rgb images to get the corresponding mask images.
  - `handle_data_annotation.py` generates a GUI for the manual annotation easily.
  - `handle_data_augmentation.py` uses Flip/Resize/Translate/Crop to enlarge the dataset.
  - `handle_grasp_unlock_dataset.py` shows the class of our dataset for the model.
- Model 
  - `handle_grasp_unlock_model.py` shows the architecture of our model.
  - `train.py` training code
- Results
  - `eval.py` evaluation code
  - `test.py` test code
  - `get_dxdyR.py` is the GUM API to get the parameters.

More details can be checked in [gum.py](../gum.py)

## Acknowledgements
Thank [Feiyu Chen](https://github.com/felixchenfy) for providing the initial [code](https://github.com/felixchenfy/ros_detect_planes_from_depth_img).
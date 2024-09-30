import os
from PIL import Image
import sys
root_dir = "../"
sys.path.append(root_dir)
from utils.lib_io import *

# ## rename files
root_dir = r'./data/lever/original'
# rename_files_sequentially(folder=root_dir,digits=3)

# ## new folder
# new_root_dir = r'E:\realman-robot\open_door\data\lever_handle_2'
# mkdir(new_root_dir)

# ## get filenames
names = get_filenames(folder=root_dir,is_base_name=False,filter='HEIC')

# ## delete_by_size
# delete_by_size(root_dir,filter='jpg',img_w_max=5000,img_w_min=640,img_h_max=5000,img_h_min=640)

# ## resize
# for name in names:
#     resize_img_equal_ratio(name, 1280, 1280, output_path=name)

# # ## HEIC to png
for name in names:
    jpg_path = f"{root_dir}/{os.path.basename(name.replace('HEIC','jpg'))}"
    os.remove(f"{root_dir}/{os.path.basename(name.replace('HEIC','png'))}")
    convert_heic_to_jpg_imagemagick(heic_path=name,jpg_path=jpg_path)
    resize_img(jpg_path, width=1280, height=720, output_path=None)
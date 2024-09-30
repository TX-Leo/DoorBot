import os
import sys
import glob
import yaml
import csv
import simplejson
import time
import numpy as np
from PIL import Image
import cv2
import shutil

def makedirs(output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

def read_yaml_file(file_path, is_convert_dict_to_class=True):
    with open(file_path, 'r') as stream:
        data = yaml.safe_load(stream)
    if is_convert_dict_to_class:
        data = Config(data)
    return data

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = simplejson.load(f)
    return data

def read_csv_file(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            data.append(row)
    data_np = np.array(data, dtype=np.float32)
    return data_np

def list_files_and_folders(directory):
    folders = []
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item_path)
        elif os.path.isfile(item_path):
            files.append(item_path)
    return folders, files

def list_files_and_folders_recursively(directory):
    folders = []
    files = []
    def walk_directory(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                folders.append(item_path)
                walk_directory(item_path)  # Recursively call for subfolders
            elif os.path.isfile(item_path):
                files.append(item_path)
    walk_directory(directory)
    return folders, files

def select_files_by_name(files,suffix='log'):
    selected_files = []
    for file in files:
        if file.endswith(suffix):
            selected_files.append(file)
    return selected_files

def get_filenames(folder, is_base_name=False, filter=None): # filter: 'png' ,'txt' ...
    ''' Get all filenames under the specific folder. 
    e.g.:
        full name: data/rgb/000001.png
        base name: 000001.png 
    '''
    full_names = sorted(glob.glob(folder + "/*"))
    
    if is_base_name:
        base_names = [name.split("/")[-1] for name in full_names]
        if filter:
            if isinstance(filter,str):
                base_names = [name for name in base_names if name.endswith(filter)]
                return base_names
            elif isinstance(filter,list):
                new_base_names = []
                for name in base_names:
                    for f in filter:
                        if name.endswith(f):
                            new_base_names.append(name)
                            break
                return new_base_names
    else:
        if filter:
            if isinstance(filter,str):
                full_names = [name for name in full_names if name.endswith(filter)]
                return full_names
            elif isinstance(filter,list):
                new_full_names = []
                for name in full_names:
                    for f in filter:
                        if name.endswith(f):
                            new_full_names.append(name)
                            break
                return new_full_names

def rename_files_sequentially(folder, digits=None):
    """Renames all files in a folder sequentially.

    Args:
        folder (str): The path to the folder containing the files.
        digits (int, optional): The number of digits for zero-padding 
                                 the file names. If None, no padding is applied. 
                                 Defaults to None.
    """
    files = sorted(os.listdir(folder))
    for i, file in enumerate(files):
        old_path = os.path.join(folder, file)
        extension = os.path.splitext(file)[1]

        if digits is not None:
            new_file = f"{i:0{digits}}{extension}"  # Apply zero-padding
        else:
            new_file = f"{i}{extension}"  # No padding

        new_path = os.path.join(folder, new_file)
        os.rename(old_path, new_path)
        # print(f"Renamed '{file}' to '{new_file}'")

def copy_files_with_sequential_names(file_list, destination_folder):
  """Copies files from a list to a destination folder with sequential numbering.

  Args:
    file_list: A list of file paths to copy.
    destination_folder: The path to the destination folder.
  """

  for i, source_file in enumerate(file_list):
    # Get the file extension (e.g., '.jpg', '.png')
    file_extension = os.path.splitext(source_file)[1]  
    # Create the new file name with sequential numbering (000, 001, ...)
    new_file_name = f"{i:03d}{file_extension}"  
    # Construct the full destination path
    destination_path = os.path.join(destination_folder, new_file_name)
    # Copy the file using shutil.copy()
    shutil.copy(source_file, destination_path)  

def convert_heic_to_png_pyheif(heic_path, png_path):
    """Converts a HEIC file to PNG format.

    Args:
        heic_path (str): The path to the HEIC file.
        png_path (str): The path to save the converted PNG file.
    """
    import pyheif
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    image.save(png_path, "PNG")

    import subprocess

def convert_heic_to_png_imagemagick(heic_path, png_path):
    """Converts a HEIC file to PNG using ImageMagick.

    Args:
        heic_path (str): The path to the HEIC file.
        png_path (str): The path to save the PNG file.
    """
    ## need to install ImageMagick Application
    import os
    os.environ['PATH'] += os.pathsep + r"D:\ImageMagic\ImageMagick-7.1.1-Q16-HDRI"
    # print(os.environ['PATH'])
    
    import subprocess
    subprocess.run(["magick", "convert", heic_path, png_path])

def convert_heic_to_jpg_imagemagick(heic_path, jpg_path):
    """Converts a HEIC file to jpg using ImageMagick.

    Args:
        heic_path (str): The path to the HEIC file.
        jpg_path (str): The path to save the jpg file.
    """
    ## need to install ImageMagick Application
    import os
    os.environ['PATH'] += os.pathsep + r"D:\ImageMagic\ImageMagick-7.1.1-Q16-HDRI"
    # print(os.environ['PATH'])
    
    import subprocess
    subprocess.run(["magick", "convert", heic_path, jpg_path])

def resize_img(image_path, width, height, output_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Unable to read image from {image_path}")
        return False
    # Resize the image
    resized_img = cv2.resize(img, (width, height))
    # Determine the output path
    if output_path is None:
        output_path = image_path
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    # Save the resized image
    success = cv2.imwrite(output_path, resized_img)
    if success:
        print(f"Resized image saved to: {output_path}")
    else:
        print(f"Error: Failed to save resized image to {output_path}")
    return success

def resize_img_equal_ratio(image_path, max_w, max_h, output_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Unable to read image from {image_path}")
        return False

    img_h, img_w = img.shape[:2]
    
    w_ratio = img_w/max_w
    h_ratio = img_h/max_h
    
    if w_ratio > 1 or h_ratio > 1:
        ratio = max(w_ratio,h_ratio)
        img = cv2.resize(img, (int(img_w/ratio), int(img_h/ratio)))

    # Determine the output path
    if output_path is None:
        output_path = image_path
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    # Save the resized image
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"Resized image saved to: {output_path}")
    else:
        print(f"Error: Failed to save resized image to {output_path}")
    return success

class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, (dict)):
                setattr(self, key, self.__class__(value))
            else:
                setattr(self, key, value)

def getch_win():
    import msvcrt
    char = msvcrt.getch()
    # special char
    if char == b'\xe0':
        return {
            b'U': "up",
            b'P': "down",
            b'K': "left",
            b'M': "right",
        }.get(char, None)
    # normal char
    else:
        return char.decode('utf-8')

def getch_linux():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char
    
def getch(if_p=False):
    if os.name == 'nt':  # Windows
       char = getch_win()
    else:  # Linux
       char = getch_linux()
    if if_p:
        print(f'char: {char}')
    return char

def mkdir(dir):
    if not os.path.exists(dir): # Check if the full path (file or directory) exists
        os.makedirs(dir) 

def mkfile(path):
    dir = os.path.dirname(path)
    mkdir(dir)

def delete_by_size(root_dir,filter='png',img_w_max=5000,img_w_min=640,img_h_max=5000,img_h_min=640):
    names = get_filenames(folder=root_dir,is_base_name=False,filter=filter)
    for name in names:
        img = Image.open(name)
        img_w, img_h = img.size
        if img_w > img_w_max or img_w < img_w_min or img_h > img_h_max or img_h < img_h_min:
            print(f'delete {name}')
            try:
                os.chmod(name, 0o777) 
                os.remove(name)
                print(f"delete {name}")
            except PermissionError:
                print(f"{name} PermissionError")
            except FileNotFoundError:
                print(f"Not Found: {name}")

def replace_backslash(path):
    return path.replace("\\", "/")


if __name__ == "__main__":
    root_dir = r'E:\realman-robot\open_door\data\lever_handle_2'
    # rename_files_sequentially(folder=root_dir,digits=3)
    
    new_root_dir = r'E:\realman-robot\open_door\data\lever_handle_2'
    if not os.path.exists(new_root_dir):
        os.makedirs(new_root_dir)

    names = get_filenames(folder=root_dir,is_base_name=False,filter='HEIC')
    
    for name in names:
        png_path = f"{new_root_dir}/{os.path.basename(name.replace('HEIC','png'))}"
        convert_heic_to_png_imagemagick(heic_path=name,png_path=png_path)
        resize_and_save_image(png_path, width=1280, height=720, output_path=None)
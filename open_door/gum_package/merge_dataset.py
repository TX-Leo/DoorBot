'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-22 13:57:08
Version: v1
File: 
Brief: 
'''
import os
import shutil

def merge_datasets(folder1, folder2):
    """
    Merges two datasets, prioritizing files from folder1,
    and renames the combined dataset sequentially in folder1.
    
    Args:
    folder1 (str): Path to the first dataset folder.
    folder2 (str): Path to the second dataset folder.
    """

    # Create a list of existing files in folder1 with only the base filename (no extension)
    # existing_files1 = [os.path.splitext(f)[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    existing_files1 = [os.path.splitext(f)[0] for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f)) and f.lower().endswith('.jpg')]
    existing_files2 = [os.path.splitext(f)[0] for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f)) and f.lower().endswith('.jpg')]

    # Function to get the next available filename
    def get_next_filename(index):
        while True:
            filename = f"{index:03d}"
            # print(f'filename:{filename}')
            if filename not in existing_files1:
                existing_files1.append(filename)  # Mark this filename as used
                return filename
            index += 1

    index = 0 
    for filename in existing_files2[:]:  # Iterate over a copy to avoid modifying while iterating
        new_filename = get_next_filename(index)
        print(f'new_filename: {new_filename}')
        for ext in ['.HEIC','.jpg', '.json', '.txt', '_annotated.png', '_mask.png', '_vis.png']:
            source_path = os.path.join(folder2, filename + ext)
            # print(f'source_path: {source_path}')
            if os.path.exists(source_path):
                dest_path = os.path.join(folder1, new_filename + ext)
                # print(f'dest_path: {dest_path}')
                shutil.move(source_path, dest_path)
        index += 1  # Increment index after processing all extensions for a filename 

    # Clear and remove folder2
    # shutil.rmtree(folder2, ignore_errors=True) 

if __name__ == "__main__":
    folder1 = './data/lever/'
    folder2 = './data/lever2/'
    merge_datasets(folder1, folder2)
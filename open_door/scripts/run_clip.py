import sys
root_dir = "../"
sys.path.append(root_dir)

from utils.lib_clip import CLIP

_clip = CLIP.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_clip.yaml')
rgb_img = f'{root_dir}/example_img/ll_example_img1.png'
_clip.clip_detection(rgb_img,text_prompt=["door that is closed", "door that is open"],if_p=True)
import sys
root_dir = "../"
sys.path.append(root_dir)

from utils.lib_gemini import GEMINI
from prompt import *

gemini = GEMINI.init_from_yaml(f'{root_dir}/cfg/cfg_gemini.yaml')

## hl
example_img1 = f'{root_dir}/example_img/ll_example_img1.png'
last_pmt_action = 'Premove'
last_pmt_error = 'SUCCESS'
hl_prompt = gen_hl_prompt(last_pmt_action,last_pmt_error)
gemini.text_img_to_text(text=hl_prompt,img_path=example_img1,if_p=True)

## ll
example_param1 = [-42.79, 9.61, 94.04]
example_param2 = [-45.11, 19.86, 2.28]
example_img1 = f'{root_dir}/example_img/ll_example_img1.png'
example_img2 = f'{root_dir}/example_img/ll_example_img2.png'
img_path = f'{root_dir}/example_img/ll_example_img3.png'
ll_prompt = gen_ll_prompt(example_param1,example_param2)
gemini.text_imgs_to_text(ll_prompt,img_paths=[example_img1,example_img2,img_path],if_p=True)

## detection
example_img1 = f'{root_dir}/example_img/ll_example_img1.png'
gemini.gemini_detection(img_path=example_img1,text_prompt=["door that is closed", "door that is open"],if_p=True)
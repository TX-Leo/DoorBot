'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-08-10 19:17:57
Version: v1
File: 
Brief: 
'''
import sys
root_dir = "../"
sys.path.append(root_dir)

import time 

from utils.lib_clip import CLIP
from utils.lib_gemini import GEMINI

## init clip and gemini
_clip = CLIP.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_clip.yaml')
gemini = GEMINI.init_from_yaml(f'{root_dir}/cfg/cfg_gemini.yaml')

## get all images
directory = f'{root_dir}/images/all_types/'
lever_pull = [f'{directory}/lever1.png'f'{directory}/lever2.png',f'{directory}/lever3.png',f'{directory}/lever4.png']
lever_push = [f'{directory}/lever5.png']
doorknob_pull = [f'{directory}/doorknob2.png',f'{directory}/doorknob4.png']
doorknob_push = [f'{directory}/doorknob1.p  ng',f'{directory}/doorknob3.png',f'{directory}/doorknob5.png']
doorknob_push = [f'{directory}/doorknob3.png',f'{directory}/doorknob5.png']
crossbar_pull = [f'{directory}/crossbar1.png',f'{directory}/crossbar2.png',f'{directory}/crossbar4.png',f'{directory}/crossbar5.png']
crossbar_push = [f'{directory}/crossbar3.png']
drawer_pull = [f'{directory}/drawer1.png',f'{directory}/drawer2.png',f'{directory}/drawer3.png',f'{directory}/drawer4.png',f'{directory}/drawer5.png']
drawer_push = []

pull = lever_push + doorknob_pull + crossbar_pull + drawer_pull
push = lever_push + doorknob_push + crossbar_push + drawer_push

## loop through pull
clip_success_pull = 0
gemini_success_pull = 0
num = 0
for img in pull:
    num += 1
    print(f'***************************************************')
    print(f'pull_num_{num} ... ')

    text_prompt=["door/drawer/cabinet/fridge/microwave that needs to be pulled","door/drawer/cabinet/fridge/microwave that needs to be pushed"]
    
    # clip
    clip_result,clip_text,clip_probs = _clip.clip_detection(img,text_prompt=text_prompt,if_p=True)
    if clip_result == 0:
        clip_success_pull += 1
    
    # gemini
    gemini_result,gemini_text,gemini_probs = gemini.gemini_detection(img,text_prompt=text_prompt,if_p=True)
    if gemini_result == 0:
        gemini_success_pull += 1

    print(f'clip_success_pull: {clip_success_pull}')
    print(f'gemini_success_pull: {gemini_success_pull}')

    time.sleep(10)

print(f'clip success rate (pull): {clip_success_pull} / {len(pull)}')
print(f'gemini success rate (pull): {gemini_success_pull} / {len(pull)}')

## loop through push
clip_success_push = 0
gemini_success_push = 0
num = 0
for img in push:
    num += 1
    print(f'***************************************************')
    print(f'push_num_{num} ... ')

    text_prompt=["door/drawer/cabinet/fridge/microwave that needs to be pulled","door/drawer/cabinet/fridge/microwave that needs to be pushed"]
    
    # clip
    clip_result,clip_text,clip_probs = _clip.clip_detection(img,text_prompt=text_prompt,if_p=True)
    if clip_result == 1:
        clip_success_push += 1
    
    # gemini
    gemini_result,gemini_text,gemini_probs = gemini.gemini_detection(img,text_prompt=text_prompt,if_p=True)
    if gemini_result == 1:
        gemini_success_push += 1
    
    print(f'clip_success_push: {clip_success_push}')
    print(f'gemini_success_push: {gemini_success_push}')

    time.sleep(10)


print(f'clip success rate (push): {clip_success_push} / {len(push)}')
print(f'gemini success rate (push): {gemini_success_push} / {len(push)}')


## all success rate

clip_success_all = clip_success_pull + clip_success_push
gemini_success_all = gemini_success_pull + gemini_success_push
total_img = len(pull) + len(push)

print(f'clip success rate (all): {clip_success_all} / {total_img}')
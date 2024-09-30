'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-08-10 19:17:57
Version: v1
File: 
Brief: 
'''
import sys
root_dir = "../../"
sys.path.append(root_dir)

import re
from utils.lib_io import *

directory = f'{root_dir}/trajectory/'
folders, files = list_files_and_folders_recursively(directory)
log_files = select_files_by_name(files, suffix='log')

# print(f'log_files: {log_files}')

num = 0

all_clip_results = {}
all_gemini_results = {}

for log_file in log_files:
    num += 1
    clip_results = {}  
    gemini_results = {}  

    with open(log_file, 'r') as f:
        current_action = None
        for line in f:
            action_match = re.search(r"\[(.*?)\]", line)
            if action_match:
                current_action = action_match.group(1) 

            if "CLIP Result" in line and current_action:
                if current_action not in clip_results:
                    clip_results[current_action] = {'total': 0, 'success': 0}

                clip_results[current_action]['total'] += 1
                match = re.search(r"clip_success:\s*(True|False)", line)
                if match and match.group(1) == "True":
                    clip_results[current_action]['success'] += 1
            
            if "gemini Result" in line and current_action:
                if current_action not in gemini_results:
                    gemini_results[current_action] = {'total': 0, 'success': 0}

                gemini_results[current_action]['total'] += 1
                match = re.search(r"gemini_success:\s*(True|False)", line)
                if match and match.group(1) == "True":
                    gemini_results[current_action]['success'] += 1

    # Update the all_clip_results dictionary
    for action, results in clip_results.items():
        if action not in all_clip_results:
            all_clip_results[action] = {'total': 0, 'success': 0}
        all_clip_results[action]['total'] += results['total']
        all_clip_results[action]['success'] += results['success']
    
    # Update the all_gemini_results dictionary
    for action, results in gemini_results.items():
        if action not in all_gemini_results:
            all_gemini_results[action] = {'total': 0, 'success': 0}
        all_gemini_results[action]['total'] += results['total']
        all_gemini_results[action]['success'] += results['success']


all_clip_success_rate = {}
all_gemini_success_rate = {}
# Calculate success rates for all actions based on accumulated data
for action, results in all_clip_results.items():
    if results['total'] > 0:
        all_clip_success_rate[action] = results['success'] / results['total'] * 100
    else:
        all_clip_success_rate[action] = 0.0
for action, results in all_gemini_results.items():
    if results['total'] > 0:
        all_gemini_success_rate[action] = results['success'] / results['total'] * 100
    else:
        all_gemini_success_rate[action] = 0.0

print(f'log_num: {num} all_clip_results: {all_clip_results}')
print(f'log_num: {num} all_clip_success_rate: {all_clip_success_rate}')
print(f'log_num: {num} all_gemini_results: {all_gemini_results}')
print(f'log_num: {num} all_gemini_success_rate: {all_gemini_success_rate}')
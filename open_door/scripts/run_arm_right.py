'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-08-10 17:40:15
Version: v1
File: 
Brief: 
'''
import sys
root_dir = "../"
sys.path.append(root_dir)

from arm import Arm

arm = Arm.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_arm_right.yaml')

print(arm)

arm.run()

arm.disconnect()
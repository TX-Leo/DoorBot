'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-18 13:02:26
Version: v1
File: 
Brief: 
'''
import sys
root_dir = "../"
sys.path.append(root_dir)

import numpy as np

from arm import Arm

arm = Arm.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_arm_right.yaml')
print(arm)

grasp_dmp_refer_tjt_path = f'{root_dir}/cfg/grasp_dmp_refer_tjt_right.csv'
grasp_dmp_middle_points = [95]
target2base_xyzrxryrz = np.array([0.74481064,-0.24952034,-0.15857617,-1.52667011,-1.55602563,-1.78504951])

tag = arm.move_handle_dmp(pos=target2base_xyzrxryrz,dmp_refer_tjt_path=grasp_dmp_refer_tjt_path,dmp_middle_points=grasp_dmp_middle_points,save_dir=f'./dmp/',if_p=True,if_planb=True)
               
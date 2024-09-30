import sys
root_dir = "../"
sys.path.append(root_dir)

from arm import Arm

arm = Arm.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_arm_left.yaml')

arm.run()
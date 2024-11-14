import sys
root_dir = "../"
sys.path.append(root_dir)

from arm import Arm

arm = Arm.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_arm_left.yaml')

arm.run()

# # arm.control_gripper(open_value=800) # 0-1000 

# # detection part
# pos = dtsam.detect() # Detic + SAM
x,y = 10, 30
z = 100 #　depth image

# ransac
rx,ry,rz = 0.1,0.2,0.3

# translate to base coordinates
pos = arm.target2cam_xyzrpy_to_target2base_xyzrpy(target2cam_xyzrpy=[x,y,z,rx,ry,rz])

arm.move_p(pos,vel)

arm.control_gripper(open_value=800)

ret = arm.get_gripper_grasp_return()

if ret != 2:
    print("Gripper not grasped")
'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-03 09:01:25
Version: v1
File: 
Brief: 
'''
import csv
import sys
import time
import os
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.lib_math import *
from utils.lib_io import *

from arm_package.robotic_arm import Arm as ArmBase
from dmp import DMP

VZ_SPEED = 0.025 # 2.5cm/s
VZ_SPEED_DEGREE = 14 # 14°/s
VYAW_SPEED_DEGREE = 35 # 35°/s
VYAW_SPEED_RADIAN = 25*np.pi/180

## for dh_gripper
ADDRESS_INIT_GRIPPER = int(0x0100)
ADDRESS_SET_FORCE = int(0x0101)
ADDRESS_SET_POS = int(0X0103)
ADDRESS_SET_VEL = int(0X0104)
ADDRESS_GET_GRIPPER_INIT_RETURN = int(0x0200)
ADDRESS_GET_GRIPPER_GRASP_RETURN = int(0x0201)
ADDRESS_GET_GRIPPER_POS = int(0x0202)
GRIPPER_VOLTAGE = 3
GRIPPER_PORT = 1
GRIPPER_BAUDRATE = 115200
GRIPPER_DEVICE = 1

class Arm():
    def __init__(self,root_dir='./',host_ip='192.168.10.19',host_port=8080,cam2base_H_path='cfg/cam2base_H_right.csv',tool_frame='dh3',home_state=[0,0,0,0,0,0,0],middle_state=[0,0,0,0,0,0,0],arm_vel=15,if_gripper=False,gripper_force=30,gripper_start_pos=1000,gripper_vel=50):
        self.root_dir = root_dir
        self.host_ip = host_ip
        self.host_port = host_port
        self.tool_frame = tool_frame
        self.home_state = home_state
        self.middle_state = middle_state
        self.cam2base_H_path = cam2base_H_path
        self.cam2base_H = read_csv_file(f'{self.root_dir}/{self.cam2base_H_path}')
        self.arm_vel = arm_vel

        self.if_gripper = if_gripper
        self.gripper_force = gripper_force
        self.gripper_start_pos = gripper_start_pos
        self.gripper_vel = gripper_vel

        self.connect()
        if if_gripper:
            self.connect_gripper(gripper_force,gripper_start_pos,gripper_vel)
        self.home()
    
    @classmethod
    def init_from_yaml(cls,root_dir='./',cfg_path='cfg/cfg_arm_right.yaml'):
        cfg = read_yaml_file(f'{root_dir}/{cfg_path}', is_convert_dict_to_class=True)
        return cls(root_dir,cfg.host_ip,cfg.host_port,cfg.cam2base_H_path,cfg.tool_frame,cfg.home_state,cfg.middle_state,cfg.arm_vel,cfg.if_gripper,cfg.gripper_force,cfg.gripper_start_pos,cfg.gripper_vel)

    def __str__(self):
        # self.get_j()
        # self.get_p()
        # self.get_v()
        # self.get_c()
        # self.get_api_version()
        # self.get_current_tool_frame(if_p=True)
        return ''

    def connect(self):
        print('==========\nArm Connecting...')
        self.arm = ArmBase(self.host_ip,self.host_port)
        self.change_tool_frame(self.tool_frame)
        print('Arm Connected\n==========')

    def home(self):
        if self.if_gripper:
            self.control_gripper(open_value=1000)
        time.sleep(1)
        self.go_home()

    def disconnect(self):
        self.arm.Arm_Socket_Close()
    
    def get_api_version(self):
        self.arm.API_Version()

    def connect_gripper(self,force=30,start_pos=1000,vel=50):
        print('==========\nGripper Connecting...')
        tag = self.arm.Set_Tool_Voltage(type=GRIPPER_VOLTAGE,block=True)
        tag = self.arm.Set_Modbus_Mode(port=GRIPPER_PORT, baudrate=GRIPPER_BAUDRATE, timeout=2, block=True)
        tag = self.arm.Write_Single_Register(port=GRIPPER_PORT, address=ADDRESS_INIT_GRIPPER, data=1, device=GRIPPER_DEVICE, block=True)
        tag = self.arm.Write_Single_Register(port=GRIPPER_PORT, address=ADDRESS_SET_FORCE, data=force, device=GRIPPER_DEVICE, block=True)
        tag = self.arm.Write_Single_Register(port=GRIPPER_PORT, address=ADDRESS_SET_POS, data=start_pos, device=GRIPPER_DEVICE, block=True)
        tag = self.arm.Write_Single_Register(port=GRIPPER_PORT, address=ADDRESS_SET_VEL, data=vel, device=GRIPPER_DEVICE, block=True)
        tag, value = self.arm.Get_Read_Input_Registers(port=GRIPPER_PORT, address=ADDRESS_GET_GRIPPER_INIT_RETURN, device=GRIPPER_DEVICE)
        if value != 1: # 0: not init. 1: init is successful. 2: initializing
            print(f'[Arm Info] Init Failed: {value}!!!!!!! Re-init Gripper...')
            time.sleep(0.5)
            self.connect_gripper()
        print('Gripper Connected\n==========')
        return tag

    def control_gripper(self,open_value,block=True):
        tag = self.arm.Write_Single_Register(port=GRIPPER_PORT, address=ADDRESS_SET_POS, data=open_value, device=GRIPPER_DEVICE, block=block)
        return tag
    
    def get_gripper_grasp_return(self,if_p=False):
        grasping_return = {0:"Gripper Moving",1:"No Objects Grasping",2:"Objects Grasping",3:"Objects Dropped After Grasping"}
        tag, value = self.arm.Get_Read_Input_Registers(port=GRIPPER_PORT, address=ADDRESS_GET_GRIPPER_GRASP_RETURN, device=GRIPPER_DEVICE)
        if if_p:
            print(f'[Gripper INFO] Grasping Detection Result: {value}  INFO: {grasping_return[value]}')
        return value # 0 for moving; 1 for detecting no objects grasping; 2 for detecting objects grasping; 3 for detecting objecting dropped after detecting grasping

    def get_gripper_pos(self,if_p=False):
        tag, value = self.arm.Get_Read_Input_Registers(port=GRIPPER_PORT, address=ADDRESS_GET_GRIPPER_POS, device=GRIPPER_DEVICE)
        if if_p:
            print(f'[Gripper INFO] Gripper Pos: {value}')
        return value # 0-1000

    def go_home(self,vel=None,block=True):
        if not vel:
            vel = self.arm_vel
        self.move_j(joint=self.home_state,vel=vel,block=block)
    
    def get_p(self,if_p=False):
        pose = self.arm.Get_Current_Pose()
        if if_p:
            print(f'[Arm INFO]: - {self.get_p.__name__}: {pose}')
        return pose

    def get_j(self,if_p=False):
        tag,joint = self.arm.Get_Joint_Degree()
        if if_p:
            print(f'[Arm INFO]: - {self.get_j.__name__}: {joint}')
        return joint

    def get_v(self,if_p=False):
        tag, voltage = self.arm.Get_Joint_Voltage()
        if if_p:
            print(f'[Arm INFO]: - {self.get_v.__name__}: {voltage}')
        return voltage

    def get_c(self,if_p=False):
        tag, current = self.arm.Get_Joint_Current()
        if if_p:
            print(f'[Arm INFO]: - {self.get_c.__name__}: {current}')
        return current
    
    def move_j(self,joint,vel=None,trajectory_connect=0, r=0, block=True, if_p=False):
        if not vel:
            vel = self.arm_vel
        tag = self.arm.Movej_Cmd(joint, vel, trajectory_connect, r, block) 
        if if_p:
            print(f'[Arm INFO]: - {self.move_j.__name__}: {tag}')
        return tag

    def move_p(self,pos,vel=None,trajectory_connect=0, r=0, block=True, if_p=False):
        if not vel:
            vel = self.arm_vel
        tag = self.arm.Movej_P_Cmd(pos, vel, trajectory_connect, r, block)
        if if_p:
            print(f'[Arm INFO]: - {self.move_p.__name__}: {tag}')
        return tag

    def move_handle_middle(self,pos,middle=None,vel=None,if_p=False):
        if not vel:
            vel = self.arm_vel
        
        if not middle:
            tag1 = self.move_j(joint=self.middle_state,vel=vel,if_p=if_p)
        else:
            tag1 = self.move_p(pos=middle,vel=vel,if_p=if_p)
        time.sleep(1)
        tag2 = self.move_p(pos=pos,vel=vel,if_p=if_p)
        
        return tag1 !=0 or tag2 != 0

    def move_handle_dmp(self,pos,dmp_refer_tjt_path,dmp_middle_points,vel=None,save_dir=None,if_p=False,if_planb=False):
        if not vel:
            vel = self.arm_vel
        
        ## init dmp
        self.dmp = DMP(f'{self.root_dir}/{dmp_refer_tjt_path}')
        self.dmp_middle_points = dmp_middle_points
        if save_dir:
            mkdir(save_dir)
            self.new_tjt = self.dmp.gen_new_tjt(initial_pos=self.get_p(),goal_pos=pos,if_save=True,tjt_save_path=f'{save_dir}/refer_tjt.csv',img_save_path=f'{save_dir}/dmp.png',show=False)
        else:
            self.new_tjt = self.dmp.gen_new_tjt(initial_pos=self.get_p(),goal_pos=pos,if_save=False)
        # # start moving
        for middle_point in self.dmp_middle_points:
            for num in range(middle_point-5,middle_point+5,3):
                self.middle_pose = self.dmp.get_middle_pose(tjt=self.new_tjt,num=num)
                tag1 = self.move_p(pos=self.middle_pose,vel=vel,if_p=if_p)
                if tag1 == 0:
                    break
            if tag1 !=0:
                break
        
        # # to the goal pos
        if tag1 == 0:
            tag2 = self.move_p(pos=pos,vel=vel,if_p=if_p)
        else:
            tag2 = -1
        
        # # plan b
        if if_planb:
            if tag2 != 0:
                if tag1 != 0:
                    tag1 = self.move_j(joint=self.middle_state,vel=vel,if_p=if_p)
                tag2 = self.move_p(pos=pos,vel=vel,if_p=if_p)
        
        return tag1 !=0 or tag2 != 0

    def move_poses(self,poses,vel=None,trajectory_connect=1,if_p=False):
        if not vel:
            vel = self.arm_vel
        for pos in poses:
            self.move_p(pos=pos, vel=vel,trajectory_connect=trajectory_connect, r=0, block=True,if_p=if_p)

    def move_l(self,pos,vel=None,trajectory_connect=0, r=0, block=True, if_p=False):
        if not vel:
            vel = self.arm_vel
        tag = self.arm.Movel_Cmd(pos, vel, trajectory_connect, r, block)
        if if_p:
            print(f'[Arm INFO]: - {self.move_l.__name__}: {tag}')
        return tag

    def move_c(self):
        # Movec_Cmd(self, pose_via, pose_to, v, loop, trajectory_connect, r=0, block=True):
        pass

    def move_j_with_input(self):
        while True:
            pose_input = input("Enter the pose (joint angles): ")
            if pose_input == 'q':
                break
            pose_list = [float(num) for num in pose_input.split(',')]
            if len(pose_list) == 7:
                pose_list.append(10)  # Default velocity
            joint_angles = pose_list[:7]
            velocity = int(pose_list[7])
            print(f'joints: {joint_angles}')
            print(f'velocity: {velocity}')
            self.move_j(joint_angles, velocity)

    def move_p_with_input(self):
        while True:
            pose_input = input("Enter the pose: ")
            if pose_input == 'q':
                break
            pose_list = [float(num) for num in pose_input.split(',')]
            if len(pose_list) == 6:
                pose_list.append(10)  # Default velocity
            pose = pose_list[:6]
            # for i in range(3):
            #     pose[i] = pose[i] / 1000 # mm to m
            velocity = int(pose_list[6])
            print(f'pose: {pose}')
            print(f'velocity: {velocity}')
            self.move_p(pose, velocity)

    def move_l_with_input(self):
        while True:
            pose_input = input("Enter the pose: ")
            if pose_input == 'q':
                break
            pose_list = [float(num) for num in pose_input.split(',')]
            if len(pose_list) == 6:
                pose_list.append(10)  # Default velocity
            pose = pose_list[:6]
            # for i in range(3):
            #     pose[i] = pose[i] / 1000 # mm to m
            velocity = int(pose_list[6])
            print(f'pose: {pose}')
            print(f'velocity: {velocity}')
            self.move_l(pose, velocity)

    def move_stop(self,if_p=False):
        tag = self.arm.Move_Stop_Cmd(block=True)
        if if_p:
            print(f'[Arm Stop]: - {self.move_stop.__name__}: {tag}')

    def rotate_handle_move_teach(self, T=1.0, v=30,if_p=False):
        start_time = time.time()
        while True:
            # self.arm.Ort_Teach_Cmd(type, direction, v, block)
            self.arm.Joint_Teach_Cmd(num=7, direction=0, v=100, block=0) # 20 will convulsions
            time.sleep(0.05)
            print(f'[Time]: {time.time() - start_time}')
            if if_p:
                print(f'[Time]: {time.time() - start_time}')
            if time.time() - start_time > T:
                self.arm.Teach_Stop_Cmd()
                print(f'Arm Stop!!!')
                break
    
    def rotate_handle_move_j(self, T=1.0, execute_v=20,if_p=False):
        joint_goal = self.get_j()
        joint_goal[6] = joint_goal[6] - T*VYAW_SPEED_DEGREE
        tag = self.move_j(joint=joint_goal,vel=execute_v,if_p=if_p)
        return tag

    def rotate_handle_move_p(self, T=1.0, execute_v=20,if_p=False):
        pos_goal = self.get_p()
        pos_goal[2] = pos_goal[2] - T*VYAW_SPEED_RADIAN
        tag = self.move_p(pos=pos_goal,vel=execute_v,if_p=if_p)
        return tag

    def unlock_handle_move_teach(self, T=1.0,v=30,if_p=False):
        start_time = time.time()
        self.arm.Start_Force_Position_Move()
        num = 0
        while True:
            self.arm.Joint_Teach_Cmd(num=7, direction=0, v=v, block=0)
            self.arm.Pos_Teach_Cmd(type=2, direction=0, v=v, block=0)
            num+=1
            time.sleep(0.5)
            if if_p:
                print(f'[Time]: {time.time() - start_time}')
            if time.time() - start_time > T:
                self.arm.Teach_Stop_Cmd()
                print(f'Arm Stop!!!')
                print(f'Num: {num}')
                break

    def unlock_handle_move_j(self, T=1.0, execute_v=20,if_p=False):
        joint_goal = self.get_j()
        joint_goal[5] = joint_goal[5] - T*VZ_SPEED_DEGREE
        joint_goal[6] = joint_goal[6] - T*VYAW_SPEED_DEGREE
        tag = self.move_j(joint=joint_goal,vel=execute_v,if_p=if_p)
        return tag

    def unlock_handle_move_p(self, T=1.0, execute_v=20,if_p=False): # T: positive means counter-clockwise; negative meansclockwise
        num=1
        z_diff = -abs(T)*VZ_SPEED
        yaw_diff = -T*VYAW_SPEED_DEGREE
        for i in range(num):
            joint = self.get_j()
            joint[6] += yaw_diff*(i+1)/num
            tag1 = self.move_j(joint=joint,vel=execute_v,if_p=if_p)
            pos = self.get_p()
            pos[2] += z_diff*(i+1)/num
            tag2 = self.move_p(pos=pos,vel=execute_v,if_p=if_p)
        return tag1,tag2

    def target2cam_xyzrpy_to_target2base_xyzrpy(self,target2cam_xyzrpy):
        self.cam2base_H = read_csv_file(f'{self.root_dir}/{self.cam2base_H_path}')
        cam2base_H = self.cam2base_H
        target2cam_R = EulerAngle_to_R(np.array(target2cam_xyzrpy[3:]),rad=True)
        target2cam_t = xyz_to_t(np.array(target2cam_xyzrpy[:3]))
        target2cam_H = Rt_to_H(target2cam_R, target2cam_t)
        target2base_H = cam2base_H @ target2cam_H
        target2base_xyzrpy = H_to_xyzrpy(target2base_H,rad=True)
        return target2base_xyzrpy.tolist()

    def get_current_tool_frame(self,if_p=False):
        tag, frame = self.arm.Get_Current_Tool_Frame()
        if if_p:
            print(f'current tool frame:')
            self.arm.print_frame(frame)
        return frame

    def get_current_work_frame(self,if_p=False):
        tag, frame = self.arm.Get_Current_Work_Frame()
        if if_p:
            print(f'current tool frame:')
            self.arm.print_frame(frame)
        return frame

    def get_all_tool_frame(self,if_p=False):
        tag, tool_names, tool_len = self.arm.Get_All_Tool_Frame()
        if if_p:
            print(f'all tool names:{tool_names}')
        return tool_names

    def get_given_tool_frame(self,tool_name,if_p=False):
        tag, frame = self.arm.Get_Given_Tool_Frame(tool_name)
        if if_p:
            print(f'given tool frame:')
            self.arm.print_frame(frame)
        return frame

    def change_tool_frame(self,tool_name,if_p=False):
        self.arm.Change_Tool_Frame(tool_name)
        tag, frame = self.arm.Get_Current_Tool_Frame()
        if if_p:
            print(f'current tool frame:')
            self.arm.print_frame(frame)
        return frame

    def manual_set_tool_frame(self,tool_name,pose=[0,0,0.148,0,0,0],payload=0, x=0, y=0, z=0, block=True,if_p=True):
        self.arm.Manual_Set_Tool_Frame(tool_name, pose, payload, x, y, z, block)
        frame = self.get_given_tool_frame(tool_name,if_p=False)
        if if_p:
            print(f'set a new tool frame:')
            self.arm.print_frame(frame)
        return frame

    def ik(self,end_pos,start_joint=None,flag=1,if_p=False): # flag: 0 - quaternion; 1 - Euler Angle
        if not start_joint:
            start_joint = self.get_j(if_p=if_p)
        tag, ik_result =  self.arm.Algo_Inverse_Kinematics(start_joint, end_pos, flag)
        if if_p:
            print(f'[Arm INFO]: - {self.ik.__name__}: {tag} ik_result: {ik_result}')

    def run(self):
        num = 0
        while True:
            num += 1
            print('***************************************************************')
            user_input = input(f"[Please Input Action_{num}]: ")
            try: 
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 'getj':
                    self.get_j(if_p=True)
                elif user_input.lower() == 'getc':
                    self.get_c(if_p=True)
                elif user_input.lower() == 'getp':
                    self.get_p(if_p=True)
                elif user_input.lower() == 'toolframe':
                    self.get_current_tool_frame(if_p=True)
                elif user_input.lower() == 'movej':
                    joint = list(map(float,input("Please input joint angles: ").split(',')))
                    self.move_j(joint,if_p=True)
                elif user_input.lower() == 'movep':
                    pos = list(map(float,input("Please input pos: ").split(',')))
                    self.move_p(pos,if_p=True)
                elif user_input.lower() == 'movel':
                    pos = list(map(float,input("Please input pos: ").split(',')))
                    self.move_l(pos,if_p=True)
                elif user_input.lower() == 'gripper':
                    open_value = int(input("Please input open_value: "))
                    self.control_gripper(open_value)
                elif user_input.lower() == 'gripper0':
                    self.control_gripper(open_value=0)
                elif user_input.lower() == 'gripper1000':
                    self.control_gripper(open_value=1000)
                elif user_input.lower() == 'home':
                    self.go_home()
                elif user_input.lower() == 'middle':
                    self.move_j(self.middle_state,if_p=True)
            except Exception as e:
                print(f"ERROR TYPE: {type(e)}: {e}")
                print("Please re-input!")
            print('***************************************************************\n\n')

if __name__ =="__main__":
    ## connect
    arm_r = Arm.init_from_yaml(cfg_path='cfg/cfg_arm_right.yaml')
    print(arm_r)
    arm_l = Arm.init_from_yaml(cfg_path='cfg/cfg_arm_left.yaml')
    print(arm_l)

    arm = arm_l
    arm2 = arm_r

    arm2.run()
    
    ## get info
    # arm.get_j(if_p=True)
    # arm.get_c(if_p=True)
    # arm_l.move_j(arm_l.middle_state)
    
    ## move joint
    # joint = arm.get_j()
    # joint[6] += 90
    # arm.move_j(joint=joint,if_p=True)

    ## gripper control
    # arm.control_gripper(open_value=0)

    ## tool frame
    # arm.manual_set_tool_frame(tool_name='dh3',pose=[0,0,0.148,0,0,0],if_p=True)
    # arm.get_current_tool_frame(if_p=True)
    # arm.get_all_tool_frame(if_p=True)
    
    ## disconnect
    # arm_r.disconnect()
    # arm_l.disconnect()
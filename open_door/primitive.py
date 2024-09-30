'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-17 19:21:22
Version: v1
File: 
Brief: 
'''
import numpy as np
import time
import cv2
import sys
import os
import shutil
import json
import random
import torch
import threading
from collections import namedtuple
from matplotlib import pyplot as plt

from _primitive import _Primitive
from arm import Arm
from base import Base
from camera import Camera
from head import Head
from dtsam import DTSAM
from server import Server
from ransac import RANSAC
from dmp import DMP
from gum import GUM

from utils.lib_math import *
from utils.lib_io import *
from utils.lib_rgbd import *
from utils.lib_log import *
from utils.lib_clip import *
from utils.lib_gemini import *
from prompt import *

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time() 
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time 
        print(f"[Time] {func.__name__} execution time: {execution_time:.4f} s")
        return result
    return wrapper
class Primitive(object):
    def __init__(self,root_dir='./',tjt_num=1,type='lever',cfg_path='./cfg/cfg.yaml'):
        ## os
        self.tjt_num = tjt_num
        self.type = type
        self.action_num = 0
        self.root_dir = root_dir
        self.tjt_dir = f'{self.root_dir}/trajectory/tjt_{self.tjt_num:03d}/'
        if os.path.exists(self.tjt_dir):
            shutil.rmtree(self.tjt_dir)
        os.makedirs(self.tjt_dir)
        
        ## get cfg
        cfg = read_yaml_file(f'{root_dir}/{cfg_path}', is_convert_dict_to_class=True)
        self.cfg = cfg

        ## init two arms
        self.arm_r = Arm.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_arm_right}')
        self.arm_l = Arm.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_arm_left}')
        
        ## init camera
        self.camera = Camera.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_cam}')
        
        ## init base
        self.base = Base.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_base}')
        
        ## init head
        self.head = Head.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_head}')
        
        ## init server
        self.server = Server.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_server}')
        
        ## init ransac
        self.ransac = RANSAC(cfg_ransac=f'{root_dir}/{cfg.cfg_ransac}',cfg_cam=f'{root_dir}/{cfg.cfg_cam}')
        
        ## init dtsam
        self.dtsam = DTSAM.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_dtsam}')

        ## init handle_grasp_model
        self.gum = GUM.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_gum}')

        ## init gemini
        self.gemini = GEMINI.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_gemini}')

        ## init clip
        self.clip = CLIP.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_clip}')
        
        ## init loger
        self.logger = Logger.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_logger}')
        self.logger.log_path = f'{self.tjt_dir}/log'

        ## remote
        self.remote_python_path = cfg.remote_python_path
        self.remote_root_dir = cfg.remote_root_dir
        self.remote_img_dir = cfg.remote_img_dir
        
        ## current thresholds
        self.grasp_thresholds = [[self.cfg.threshold.grasp.l,self.cfg.threshold.grasp.h] for _ in range(6)]
        self.unlock_thresholds = [[self.cfg.threshold.unlock.l,self.cfg.threshold.unlock.h] for _ in range(6)]
        self.unlock_stop_thresholds_left = [[self.cfg.threshold.unlock.stop_l_left,self.cfg.threshold.unlock.stop_h_left] for _ in range(6)]
        self.unlock_stop_thresholds_right = [[self.cfg.threshold.unlock.stop_l_right,self.cfg.threshold.unlock.stop_h_right] for _ in range(6)]
        self.rotate_thresholds = [[self.cfg.threshold.rotate.l,self.cfg.threshold.rotate.h] for _ in range(6)]
        self.rotate_stop_thresholds_left = [[self.cfg.threshold.rotate.stop_l_left,self.cfg.threshold.rotate.stop_h_left] for _ in range(6)]
        self.rotate_stop_thresholds_right = [[self.cfg.threshold.rotate.stop_l_right,self.cfg.threshold.rotate.stop_h_right] for _ in range(6)]
        self.open_thresholds = [[self.cfg.threshold.open.l,self.cfg.threshold.open.h] for _ in range(6)]
        self.open_pull_thresholds_left = [[-np.inf,np.inf] for _ in range(6)]
        self.open_pull_thresholds_right = [[-np.inf,np.inf] for _ in range(6)]
        self.open_pull_thresholds_left[3][0] = self.cfg.threshold.open.pull_j3_l_left # pull_j3_l
        self.open_pull_thresholds_right[3][0] = self.cfg.threshold.open.pull_j3_l_right # pull_j3_l
        self.swing_thresholds = [[self.cfg.threshold.swing.l,self.cfg.threshold.swing.h] for _ in range(6)]

        ## Primitive Types
        self.HOME = self.cfg.pmts.home
        self.PREMOVE = self.cfg.pmts.premove
        self.GRASP = self.cfg.pmts.grasp
        self.UNLOCK = self.cfg.pmts.unlock
        self.ROTATE = self.cfg.pmts.rotate
        self.OPEN = self.cfg.pmts.open
        self.SWING = self.cfg.pmts.swing
        self.START = self.cfg.pmts.start
        self.FINISH = self.cfg.pmts.finish
        self.BACK = self.cfg.pmts.back
        self.CLEAR = self.cfg.pmts.clear
        self.TELEOPERATION = self.cfg.pmts.teleoperation
        self.TELEARML = self.cfg.pmts.telearml
        self.TELEARMR = self.cfg.pmts.telearmr
        self.HLVLM = self.cfg.pmts.hlvlm
        self.LLVLM = self.cfg.pmts.llvlm
        self.CAPTURE = self.cfg.pmts.capture
        
        ## Error Types
        self.SUCCESS = self.cfg.errors.success
        self.SAFETY_ISSUE = self.cfg.errors.current.safety_issue
        self.EVENT_DETECTED = self.cfg.errors.current.event_detected
        self.NO_ISSUE = self.cfg.errors.current.no_issue
        self.GRASP_SAFETY = self.cfg.errors.grasp.grasp_safety
        self.GRASP_NO_HANDLE = self.cfg.errors.grasp.grasp_no_handle
        self.GRASP_IK_FAIL = self.cfg.errors.grasp.grasp_ik_fail
        self.GRASP_MISS = self.cfg.errors.grasp.grasp_miss
        self.ROTATE_SAFETY = self.cfg.errors.rotate.rotate_safety
        self.ROTATE_MISS = self.cfg.errors.rotate.rotate_miss
        self.ROTATE_IK_FAIL = self.cfg.errors.rotate.rotate_ik_fail
        self.UNLOCK_SAFETY = self.cfg.errors.unlock.unlock_safety
        self.UNLOCK_MISS = self.cfg.errors.unlock.unlock_miss
        self.UNLOCK_IK_FAIL = self.cfg.errors.unlock.unlock_ik_fail
        self.OPEN_SAFETY = self.cfg.errors.open.open_safety
        self.OPEN_MISS = self.cfg.errors.open.open_miss
        self.OPEN_FAIL = self.cfg.errors.open.open_fail
        self.SWING_SAFETY = self.cfg.errors.swing.swing_safety

        ## init 
        self.last_pmt = _Primitive(action="START",id=self.START,ret=1,param=[0,0,0],error="START")
        self.this_pmt = _Primitive(action="START",id=self.START,ret=1,param=[0,0,0],error="START")
        self.primitives = {0:self.last_pmt.to_list()}

    def disconnect_robot(self):
        print('========== Disconnecting... ==========')
        self.camera.disconnect()
        self.arm_r.disconnect()
        self.arm_l.disconnect()
        self.base.disconnect()
        self.head.disconnect()
        self.server.disconnect()
        print('========== Disconnected ==========')

    def __str__(self):
        return ''
    
    def action2num(self,action):
        if action == "premove":
            return self.PREMOVE
        elif action == "grasp":
            return self.GRASP
        elif action == "unlock":
            return self.UNLOCK
        elif action == "rotate":
            return self.ROTATE
        elif action == "open":
            return self.OPEN
        elif action == "swing":
            return self.SWING
        elif action == "home":
            return self.HOME
        elif action == "finish":
            return self.FINISH
        elif action == "back":
            return self.BACK
        elif action == "clear":
            return self.CLEAR
        elif action == "teleoperation" or action == "tele":
            return self.TELEOPERATION
        elif action == 'telearml' or action == 'arml' or action == 'telel':
            return self.TELEARML
        elif action == 'telearmr' or action == 'armr' or action == 'teler':
            return self.TELEARMR
        elif action == 'hlvlm':
            return self.HLVLM
        elif action == 'llvlm':
            return self.LLVLM
        elif action == 'capture' or action == 'cap':
            return self.CAPTURE
        else:
            return -1

    def update(self,save_path=None):
        if save_path is None:
            save_path=f'{self.tjt_dir}/{self.action_num}.json'
        
        data = {"last_action": self.last_pmt.action,
                "last_id": self.last_pmt.id,
                "last_ret": self.last_pmt.ret,
                "last_param": self.last_pmt.param,
                "last_error": self.last_pmt.error,
                "this_action": self.this_pmt.action,
                "this_id": self.this_pmt.id,
                "this_ret": self.this_pmt.ret,
                "this_param": self.this_pmt.param,
                "this_error": self.this_pmt.error
                }
        
        with open(save_path,'w') as json_file:
            json.dump(data,json_file,indent=4)
        
        # self.this_pmt --> self.last_pmt
        for attr in ["action", "id", "ret", "param", "error"]:
            setattr(self.last_pmt, attr, getattr(self.this_pmt, attr))

        self.primitives[self.action_num] = self.this_pmt.to_list()
    
    def save_primitives(self,save_path=None):
        if save_path is None:
            save_path=f'{self.tjt_dir}/primitives.json'
        
        with open(save_path,'w') as json_file:
            json.dump(self.primitives,json_file,indent=4)

    # @time_it
    def capture(self,if_d=False,vis=False,if_update=True):
        # print('========== Image Capturing ... ==========')
        self.logger.info(f'[Camera] - Capture Image')
        if if_update:
            self.action_num += 1
            self.rgb_img_path = f'{self.tjt_dir}/{self.action_num}.png'
        if if_d:
            self.d_img_path = f'{self.tjt_dir}/{self.action_num}/d.png'
            mkfile(self.d_img_path)
            rgb_img,d_img = self.camera.capture_rgbd(rgb_save_path=self.rgb_img_path,d_save_path=self.d_img_path)
            if vis:
                save_dir = f'{self.tjt_dir}/{self.action_num}/vis/'
                mkdir(save_dir)
                vis_rgbd(d_img_path=self.d_img_path,rgb_img_path=self.rgb_img_path,save_path=f'{save_dir}/vis_rgbd.png')
                vis_d(d_img_path=self.d_img_path,save_path=f'{save_dir}/vis_d.png')
            return rgb_img,d_img 
        else:
            if if_update:
                rgb_img = self.camera.capture_rgb(rgb_save_path=self.rgb_img_path)
            else:
                rgb_img = self.camera.capture_rgb(f'{self.tjt_dir}/temp.png')
            return rgb_img
        
    def start_current_monitor_thread(self,thresholds_safety,thresholds_event=None,if_event_stop=True):
        # print(f'[thresholds_safety]: {thresholds_safety}')
        # print(f'[thresholds_event]: {thresholds_event}')
        self.current_data = {i: [] for i in range(7)}
        self.current_max = [0]*7
        self.current_min = [0]*7
        self.current_data_start_time = time.time()
        self.monitor_running = True
        self.current_monitor_thread = threading.Thread(target=self.current_monitor_loop, args=(thresholds_safety,thresholds_event,if_event_stop))
        self.current_monitor_thread.daemon = True
        self.current_monitor_thread.start()

    def current_monitor_loop(self,thresholds_safety,thresholds_event,if_event_stop):
        while self.monitor_running:
            current_check_result,current = self.check_current_safety(thresholds_safety,thresholds_event)
            if current_check_result == -1:
                # print(f"!!! SAFETY ISSUE !!!")
                self.logger.error(f'[Monitor] - SAFETY_ISSUE !!!')
                self.this_pmt.ret = self.SAFETY_ISSUE
                self.this_pmt.error = 'SAFETY_ISSUE'
                # print(f'[Now Current]: {current}')
                self.logger.info(f'[Monitor] - now_current: {current}')
                self.arm.move_stop(if_p=True)
                self.base.move_stop(if_p=True)
                break
            elif current_check_result == 0:
                # print(f"!!! Event Detected !!!")
                self.logger.info(f'[Monitor] - EVENT_DETECTED !!!')
                self.this_pmt.ret = self.EVENT_DETECTED
                self.this_pmt.error = 'EVENT_DETECTED'
                # print(f'[Now Current]: {current}')
                self.logger.info(f'[Monitor] - now_current: {current}')
                self.action_T = time.time() - self.current_data_start_time
                if if_event_stop:
                    self.arm.move_stop(if_p=True)
                    self.base.move_stop(if_p=True)
                break
            elif current_check_result == 1:
                self.this_pmt.ret = self.NO_ISSUE
                self.this_pmt.error = 'NO_ISSUE'
                self.action_T = time.time() - self.current_data_start_time
            time.sleep(0.1)

    def check_current_safety(self,thresholds_safety,thresholds_event):
        current = self.arm.get_c()
        for i in range(7):
            self.current_data[i].append(current[i])
            self.current_max[i] = max(self.current_max[i],current[i])
            self.current_min[i] = min(self.current_min[i],current[i])
        # safety issue
        for i, (min_current, max_current) in enumerate(thresholds_safety):
            if current[i] < min_current or current[i] > max_current:
                return -1,current
        # event detected
        if thresholds_event:
            for i, (min_current, max_current) in enumerate(thresholds_event):
                if current[i] < min_current or current[i] > max_current:
                    return 0,current
        # no issue
        return 1,current

    def vis_current_data(self,img_save_path=None,csv_save_path=None,show=False):
        if img_save_path is None:
            img_save_path = f'{self.tjt_dir}/{self.action_num}/haptics/current.png'
        mkfile(img_save_path)
        plt.figure()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        time_elapsed = [t - self.current_data_start_time for t in range(len(self.current_data[0]))] 
        for i in range(7):
            plt.plot(time_elapsed, self.current_data[i], label=f'Joint {i}', color=colors[i])
        plt.xlabel('Time')
        plt.ylabel('Current Value')
        plt.title('Current Data of Each Joint')
        plt.legend()
        plt.grid(True)
        plt.savefig(img_save_path)
        if show:
            plt.show()

        # save current data
        if csv_save_path is None:  
            csv_save_path = f'{self.tjt_dir}/{self.action_num}/haptics/current.csv'
        mkfile(csv_save_path)
        with open(csv_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time'] + [f'Joint {i}' for i in range(7)])
            for i in range(len(time_elapsed)):
                row = [time_elapsed[i]] + [self.current_data[j][i] for j in range(7)]
                writer.writerow(row)

    @time_it
    def premove(self,param):
        print(f'========== Premoving ... ==========')
        
        self.logger.flag(f'[Premove] - Premove Start')
        start_time = time.time()

        self.this_pmt.action = "PREMOVE"
        self.this_pmt.id = self.PREMOVE
        self.this_pmt.param = [0,0,0]

        if param:
            self.cfg.premove.offset_in_front,self.cfg.premove.d2t_coefficient,self.cfg.premove.linear_velocity = param

        rgb_img_path = f'{self.tjt_dir}/{self.action_num}/rgb.png'
        mkfile(rgb_img_path)
        shutil.copy2(self.rgb_img_path, rgb_img_path)
        d_img_path = self.d_img_path

        # print(f'RANSAC ...')
        self.logger.flag(f'[Premove] - RANSAC Start')
        last_time = time.time()
        self.normal,self.weights,self._3d_center,self._2d_center,self.mask_color = self.ransac.get_normal_server(rgb_img_path,d_img_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
        now_time = time.time()
        # print(f'[Time ransac]: {now_time-last_time} s')
        # print(f'[RANSAC Result] normal: {self.normal} weights: {self.weights}')
        self.logger.info(f'[Premove] - RANSAC Result - normal: {self.normal} weights: {self.weights} 3d_center: {self._3d_center} 2d_center: {self._2d_center} mask_color: {self.mask_color}')
        self.logger.time(f'[Premove] - RANSAC Time - {now_time-last_time} s')
        self.logger.flag(f'[Premove] - RANSAC End')

        self.logger.flag(f'[Premove] - Moving Start')
        self.move_T,self.move_distance = self.base.move_to_door(self.weights,self.cfg.premove.offset_in_front,self.cfg.premove.d2t_coefficient,self.cfg.premove.linear_velocity)
        time.sleep(2)
        # print(f'[Time Moving] move_T: {self.move_T} s')
        # print(f'[Moving Result] weights: {self.weights} offset_in_front: {self.cfg.premove.offset_in_front} d2t_coefficient: {self.cfg.premove.d2t_coefficient} linear_velocity: {self.cfg.premove.linear_velocity}')
        self.logger.info(f'[Premove] - Moving Input - weights: {self.weights} offset_in_front: {self.cfg.premove.offset_in_front} d2t_coefficient: {self.cfg.premove.d2t_coefficient} linear_velocity: {self.cfg.premove.linear_velocity}')
        self.logger.time(f'[Premove] - Moving Output - move_T: {self.move_T} s move_distance: {self.move_distance}')
        self.logger.flag(f'[Premove] - Moving End')

        self.this_pmt.ret = self.SUCCESS
        self.this_pmt.error = "NONE"
        self.logger.info(f'[Premove] - Success When Premoving')

        self.logger.info(f'[Premove] - Return - action: {self.this_pmt.action} ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')

        self.this_pmt.param = [self.cfg.premove.offset_in_front,self.cfg.premove.d2t_coefficient,self.cfg.premove.linear_velocity,self.move_T]

        self.update()
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')

        self.logger.flag(f'[Premove] - Premove End')
        end_time = time.time()
        self.logger.time(f'[Premove] - Premove Time: {end_time-start_time} s')

        print(f'========== Premove Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def grasp(self,param):
        print('========== Grasping... ==========')
        
        self.logger.flag(f'[Grasp] - Grasp Start')
        start_time = time.time()

        self.this_pmt.action = "GRASP"
        self.this_pmt.id = self.GRASP
        self.this_pmt.param = param

        ## os
        rgb_img_path = f'{self.tjt_dir}/{self.action_num}/rgb.png'
        mkfile(rgb_img_path)
        shutil.copy2(self.rgb_img_path, rgb_img_path)
        d_img_path = self.d_img_path

        ## base
        # self.grasp_base_x,self.grasp_base_y,self.grasp_base_theta = self.base.get_location()

        ## time
        now_time = time.time()

        ## dtsam
        # print('DTSAM ...')
        self.logger.flag(f'[Grasp] - DTSAM Start')
        last_time = time.time()
        if self.type == 'knob':
            self.dtsam.classes = 'doorknob'
        self.x1_2d,self.y1_2d,self.orientation,self.w,self.h,self.box = self.dtsam.get_xy_server(rgb_img_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
        now_time = time.time()
        # print(f'[Time dtsam]: {now_time-last_time} s')
        # print(f'[DTSAM Result] x1_2d: {self.x1_2d}, y1_2d: {self.y1_2d}, orientation: {self.orientation}, w: {self.w}, h: {self.h}, box: {self.box}')
        self.logger.time(f'[Grasp] - DTSAM Time - {now_time-last_time} s')
        self.logger.info(f'[Grasp] - DTSAM Result - x1_2d: {self.x1_2d}, y1_2d: {self.y1_2d}, orientation: {self.orientation}, w: {self.w}, h: {self.h}, box: {self.box}')
        self.logger.flag(f'[Grasp] - DTSAM End')

        if self.w == 0 and self.h == 0:
            self.this_pmt.ret = self.GRASP_NO_HANDLE
            self.this_pmt.error = "GRASP_NO_HANDLE"
            # print(f'[DTSAM Error] NO handle detections!!!')
            self.logger.error(f'[Grasp] - DTSAM Error - NO handle detections!!!')
        else:
            ## dx,dy,R
            if not param:
                # print('GUM ...')
                self.logger.flag(f'[Grasp] - GUM Start')
                crop_rgb_img_path = f'{os.path.dirname(rgb_img_path)}/gum/rgb_cropped.png'
                crop_image(rgb_img_path, center_x=self.x1_2d, center_y=self.y1_2d, new_w=self.gum.img_w, new_h=self.gum.img_h, save_path=crop_rgb_img_path)
                mask_path = f'{os.path.dirname(rgb_img_path)}/dtsam/center.png'
                crop_mask_path = f'{os.path.dirname(rgb_img_path)}/gum/mask_cropped.png'
                mask_image = crop_image(mask_path, center_x=self.x1_2d, center_y=self.y1_2d, new_w=self.gum.img_w, new_h=self.gum.img_h, save_path=crop_mask_path)
                
                last_time = time.time()
                self.dx,self.dy,self.R = self.gum.get_dxdyR_server(crop_rgb_img_path,crop_mask_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
                now_time = time.time()
                # print(f'[Time gum]: {now_time-last_time} s')
                # print(f'[GUM Result]: dx: {self.dx}, dy: {self.dy}, R: {self.R}')
                self.logger.time(f'[Grasp] - GUM Time - {now_time-last_time} s')
                self.logger.info(f'[Grasp] - GUM Result - dx: {self.dx}, dy: {self.dy}, R: {self.R}')
                self.logger.flag(f'[Grasp] - GUM End')
            else:
                self.dx,self.dy,self.R = param[:3]
                # print(f'dx: {self.dx}, dy: {self.dy}, R: {self.R}')
                self.logger.info(f'[Grasp] - Grasp Offset (without gum) - dx: {self.dx}, dy: {self.dy}, R: {self.R}')

            ##　grasp point 2d offset(dx,dy)
            self.x1_2d += self.dx
            self.y1_2d += self.dy

            ## for Exp2: without GUM
            # random_sign =  1 if random.random() < 0.5 else -1
            # if self.orientation == 'horizontal':
                # self.R = self.w / 2 * random_sign
            # elif self.orientation == 'vertical':
                # self.R = self.h / 2 * random_sign

            ## rotate point
            self.x2_2d,self.y2_2d,self.Ox,self.Oy = rotate_point(self.x1_2d,self.y1_2d,R=self.R,orientation=self.orientation,angle=90)
            # print(f'[p1_2d] x1_2d: {self.x1_2d}, y1_2d: {self.y1_2d}')
            # print(f'[p2_2d] x2_2d: {self.x2_2d}, y2_2d: {self.y2_2d}')
            # print(f'[center] Ox: {self.Ox}, Oy: {self.Oy}')
            self.logger.info(f'[Grasp] - p1_2d - x1_2d: {self.x1_2d}, y1_2d: {self.y1_2d}')
            self.logger.info(f'[Grasp] - p2_2d - x2_2d: {self.x2_2d}, y2_2d: {self.y2_2d}')
            self.logger.info(f'[Grasp] - center - Ox: {self.Ox}, Oy: {self.Oy}')

            ## vis
            save_path = f'{os.path.dirname(rgb_img_path)}/gum/gum.png'
            mkfile(save_path)
            vis_grasp(rgb_img_path,self.dx,self.dy,self.x1_2d,self.y1_2d,self.x2_2d,self.y2_2d,self.Ox,self.Oy,self.R,self.orientation,angle=90,save_path=save_path,show=False)

            ## determin which arm
            if self.x1_2d < self.camera.width / 2:
                self.arm = self.arm_l
                self.arm2 = self.arm_r
                self.r_l = 'left'
                self.grasp_dmp_refer_tjt_path = self.cfg.grasp.grasp_dmp_refer_tjt_path_left
                self.grasp_dmp_middle_points = self.cfg.grasp.grasp_dmp_middle_points_left
                # print(f'[Arm Choice]: LEFT')
            else:
                self.arm = self.arm_r
                self.arm2 = self.arm_l
                self.r_l = 'right'
                self.grasp_dmp_refer_tjt_path = self.cfg.grasp.grasp_dmp_refer_tjt_path_right
                self.grasp_dmp_middle_points = self.cfg.grasp.grasp_dmp_middle_points_right
                # print(f'[Arm Choice]: RIGHT')

            self.logger.info(f'[Grasp] - Arm Choice - {self.r_l}')

            ## get handle depth
            self.handle_depth = self.camera.get_depth_point(self.x1_2d,self.y1_2d,d_img=d_img_path)
            self.logger.info(f'[Grasp] - Depth Result - {self.handle_depth}')
            if self.handle_depth == 0:
                self.handle_depth = self.camera.get_handle_depth(self.x1_2d,self.y1_2d,d_img=d_img_path,orientation=self.orientation,radius=40)
                self.logger.info(f'[Grasp] - Depth Result - {self.handle_depth}')
                if self.handle_depth >10 or self.handle_depth <-10:
                    self.handle_depth = self.camera.get_depth_roi(self.x1_2d,self.y1_2d,d_img=d_img_path,radius=15,depth_threshold=0.05,valid_ratio_threshold=0.50)
                    self.logger.info(f'[Grasp] - Depth Result - {self.handle_depth}')
            # print(f'[Depth Result]: {self.handle_depth}')

            ## xy_depth 2 xyz
            self.x1_3d,self.y1_3d,self.z1_3d = self.camera.xy_depth_2_xyz(self.x1_2d,self.y1_2d,self.handle_depth)
            self.x2_3d,self.y2_3d,self.z2_3d = self.camera.xy_depth_2_xyz(self.x2_2d,self.y2_2d,self.handle_depth)
            # print(f'[xy2xyz Result] x1_3d: {self.x1_3d}, y1_3d: {self.y1_3d}, z1_3d: {self.z1_3d}')
            # print(f'[xy2xyz Result] x2_3d: {self.x2_3d}, y2_3d: {self.y2_3d}, z2_3d: {self.z2_3d}')
            self.logger.info(f'[Grasp] - xy2xyz Result - x1_3d: {self.x1_3d}, y1_w3d: {self.y1_3d}, z1_3d: {self.z1_3d}')
            self.logger.info(f'[Grasp] - xy2xyz Result - x2_3d: {self.x2_3d}, y2_3d: {self.y2_3d}, z2_3d: {self.z2_3d}')

            ## ransac
            # print('RANSAC ...')
            self.logger.flag(f'[Grasp] - RANSAC Start')
            last_time = time.time()
            self.normal,self.weights,self._3d_center,self._2d_center,self.mask_color = self.ransac.get_normal_server(rgb_img_path,d_img_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
            now_time = time.time()
            # print(f'[Time ransac]: {now_time-last_time} s')
            # print(f'[RANSAC Result] normal: {self.normal} weights: {self.weights}')
            self.logger.time(f'[Grasp] - RANSAC Time - {now_time-last_time} s')
            self.logger.info(f'[Grasp] - RANSAC Result - normal: {self.normal} weights: {self.weights} 3d_center: {self._3d_center} 2d_center: {self._2d_center} mask_color: {self.mask_color}')
            self.logger.flag(f'[Grasp] - RANSAC End')

            ## normal2rxryrz
            self.rx,self.ry,self.rz = normal2rxryrz(self.normal)
            # print(f'[normal2rxryrz Result] rx: {self.rx} ry: {self.ry} rz: {self.rz}')
            self.logger.info(f'[Grasp] - normal2rxryrz Result - rx: {self.rx} ry: {self.ry} rz: {self.rz}')

            ## p1_3d_cam_xyzrxryrz 2 p1_3d_base_xyzrxryrz
            self.p1_3d_cam_xyzrxryrz = [self.x1_3d,self.y1_3d,self.z1_3d,self.rx,self.ry,self.rz]
            self.p2_3d_cam_xyzrxryrz = [self.x2_3d,self.y2_3d,self.z2_3d,self.rx,self.ry,self.rz]
            # print(f'[p1_3d_cam_xyzrxryrz] {self.p1_3d_cam_xyzrxryrz}')
            # print(f'[p2_3d_cam_xyzrxryrz] {self.p2_3d_cam_xyzrxryrz}')
            self.logger.info(f'[Grasp] - p1_3d_cam_xyzrxryrz - {self.p1_3d_cam_xyzrxryrz}')
            self.logger.info(f'[Grasp] - p2_3d_cam_xyzrxryrz - {self.p2_3d_cam_xyzrxryrz}')
            
            self.p1_3d_base_xyzrxryrz = self.arm.target2cam_xyzrpy_to_target2base_xyzrpy(self.p1_3d_cam_xyzrxryrz)
            self.p2_3d_base_xyzrxryrz = self.arm.target2cam_xyzrpy_to_target2base_xyzrpy(self.p2_3d_cam_xyzrxryrz)
            # print(f'[p1_3d_base_xyzrxryrz] {self.p1_3d_base_xyzrxryrz}')
            # print(f'[p2_3d_base_xyzrxryrz] {self.p2_3d_base_xyzrxryrz}')
            self.logger.info(f'[Grasp] - cam2base_H - {self.arm.cam2base_H.tolist()}')
            self.logger.info(f'[Grasp] - p1_3d_base_xyzrxryrz original - {self.p1_3d_base_xyzrxryrz}')
            self.logger.info(f'[Grasp] - p2_3d_base_xyzrxryrz original - {self.p2_3d_base_xyzrxryrz}')

"""
            ## p1 offset and p2 offset
            if param:
                if len(param) == 4:
                    self.cfg.grasp.p1_depth_offset = param[3]
                elif len(param) == 5:
                    self.cfg.grasp.p1_depth_offset = param[3]
                    self.cfg.grasp.p2_depth_offset = param[4]

            def p1_offset(xyzrxryrz,r_l,orientation):
                _x,_y,_z,_rx,_ry,_rz = xyzrxryrz
                if r_l == 'right':
                    x = _x + self.cfg.grasp.p1_depth_offset
                    y = _y
                    z = _z
                    rx = _rx
                    ry = _ry+np.pi/6-np.pi/2
                    rz = _rz
                    if orientation == 'vertical':
                        ry += np.pi/2
                elif r_l == 'left':
                    x = _x
                    y = _y - self.cfg.grasp.p1_depth_offset
                    z = _z
                    rx = -1*_ry
                    ry = _rz+np.pi*2/3+np.pi/18
                    rz = -1*_rx-np.pi
                    if orientation == 'vertical':
                        rx,ry,rz = [1.0540000200271606, 0.9549999833106995, -0.6980000138282776]

                return [x,y,z,rx,ry,rz]
            
            def p2_offset(xyzrxryrz,r_l,orientation,R):
                _x,_y,_z,_rx,_ry,_rz = xyzrxryrz
                if r_l == 'right':
                    x = _x + self.cfg.grasp.p2_depth_offset
                    y = _y
                    z = _z
                    rx = _rx
                    ry = _ry
                    rz = _rz
                    if orientation == 'horizontal':
                        if R > 0:
                            ry -= np.pi/2
                        else:
                            ry += np.pi/2
                    elif orientation == 'vertical':
                        if R > 0:
                            ry += np.pi/2
                        else:
                            ry -= np.pi/2
                elif r_l == 'left':
                    x = _x
                    y = _y - self.cfg.grasp.p2_depth_offset
                    z = _z
                    rx = _rx
                    ry = _ry
                    rz = _rz
                    if orientation == 'horizontal':
                        if R > 0:
                            rx,ry,rz = [-0.7269999980926514, -0.7990000247955322, 2.127000093460083]
                        else:
                            rx,ry,rz = [0.7269999980926514, 0.7990000247955322, -1.0130000114440918]
                    elif orientation == 'vertical':
                        if R > 0:
                            rx,ry,rz = [1.2350000143051147, -0.5249999761581421, -0.09000000357627869]
                        else:
                            rx,ry,rz = [-1.2350000143051147, 0.5249999761581421, 3.049999952316284]

                return [x,y,z,rx,ry,rz]

            self.p1_3d_base_xyzrxryrz = p1_offset(self.p1_3d_base_xyzrxryrz,self.r_l,self.orientation)
            self.p2_3d_base_xyzrxryrz[3:6] = self.p1_3d_base_xyzrxryrz[3:6]
            self.p2_3d_base_xyzrxryrz = p2_offset(self.p2_3d_base_xyzrxryrz,self.r_l,self.orientation,self.R)
            # print(f'[p1_3d_base_xyzrxryrz] {self.p1_3d_base_xyzrxryrz}')
            # print(f'[p2_3d_base_xyzrxryrz] {self.p2_3d_base_xyzrxryrz}')
            self.logger.info(f'[Grasp] - p1_depth_offset - {self.cfg.grasp.p1_depth_offset}')
            self.logger.info(f'[Grasp] - p2_depth_offset - {self.cfg.grasp.p2_depth_offset}')
            self.logger.info(f'[Grasp] - p1_3d_base_xyzrxryrz after offset - {self.p1_3d_base_xyzrxryrz}')
            self.logger.info(f'[Grasp] - p2_3d_base_xyzrxryrz after offset - {self.p2_3d_base_xyzrxryrz}')
"""
            ## Current Detection Begin
            self.start_current_monitor_thread(thresholds_safety=self.grasp_thresholds)
            self.logger.info(f'[Grasp] - Current Detection Start')

            ## move to handle(DMP)
            # print(f'Moving ...')
            self.logger.flag(f'[Grasp] - Moving Start')

            ## method 1
            middle_point = self.p1_3d_base_xyzrxryrz.copy()
            front_distance = 0.2 # m, you can change the distance
            if self.r_l == 'right':
                middle_point[0] = middle_point[0] - front_distance
            elif self.r_l == 'left':
                middle_point[1] = middle_point[1] -front_distance
            # tag = self.arm.move_handle_middle(pos=self.p1_3d_base_xyzrxryrz,middle=middle_point,if_p=True)
            
            ## method 2
            tag = self.arm.move_handle_middle(pos=self.p1_3d_base_xyzrxryrz,if_p=True)
            
            ## method 3
            # tag = self.arm.move_handle_dmp(pos=self.p1_3d_base_xyzrxryrz,dmp_refer_tjt_path=self.grasp_dmp_refer_tjt_path,dmp_middle_points=self.grasp_dmp_middle_points,save_dir=f'{self.tjt_dir}/{self.action_num}/dmp/',if_p=True,if_planb=True)
            
            self.p1_joint = self.arm.get_j()
            self.logger.info(f'[Grasp] - Moving Result - Failure tag: {tag}')
            self.logger.flag(f'[Grasp] - Moving End')

            ## close gripper
            # print(f'Closing Gripper ...')
            self.logger.flag(f'[Grasp] - Gripper Closing Start')
            if not tag:
                self.arm.control_gripper(self.cfg.grasp.gripper_value)
                time.sleep(2)
            self.logger.flag(f'[Grasp] - Gipper Value: {self.cfg.grasp.gripper_value}')
            self.logger.flag(f'[Grasp] - Gripper Closing End')

            ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
            self.monitor_running = False
            self.current_monitor_thread.join()
            self.vis_current_data()
            self.logger.info(f'[Grasp] - Current Detection End')
            
            ## clip
            text_prompt=["handle with gripper grasped firmly", "handle without gripper grasped firmly"]
            clip_result,clip_text,clip_probs = self.clip.clip_detection(rgb_img=self.rgb_img_path,text_prompt=text_prompt,if_p=False)
            clip_success = (clip_result == 0 and self.arm.get_gripper_grasp_return(if_p=False) == 2) or (clip_result == 1 and self.arm.get_gripper_grasp_return(if_p=False) != 2)
            self.logger.info(f'[Grasp] - CLIP Result - clip_success: {clip_success} text_prompt: {text_prompt} clip_result: {clip_result} clip_text: {clip_text} clip_probs: {clip_probs}')
            
            ## gemini
            # gemini_result,gemini_text,gemini_probs =  self.gemini.gemini_detection(img_path=self.rgb_img_path,text_prompt=text_prompt,if_p=False)
            # gemini_success = (gemini_result == 0 and self.arm.get_gripper_grasp_return(if_p=False) == 2) or (gemini_result == 1 and self.arm.get_gripper_grasp_return(if_p=False) != 2)
            # self.logger.info(f'[Grasp] - gemini Result - gemini_success: {gemini_success} text_prompt: {text_prompt} gemini_result: {gemini_result} gemini_text: {gemini_text} gemini_probs: {gemini_probs}')

            ## update
            if self.this_pmt.ret == self.SAFETY_ISSUE:
                self.this_pmt.ret = self.GRASP_SAFETY
                self.this_pmt.error = "GRASP_SAFETY"
                self.logger.error(f'[Grasp] - GRASP_SAFETY')
            elif self.this_pmt.ret == self.NO_ISSUE or self.this_pmt.ret == self.EVENT_DETECTED:
                if tag:
                    self.this_pmt.ret = self.GRASP_IK_FAIL
                    self.this_pmt.error = "GRASP_IK_FAIL"
                    self.logger.error(f'[Grasp] - GRASP_IK_FAIL')
                elif self.arm.get_gripper_grasp_return(if_p=False) != 2:
                    self.this_pmt.ret = self.GRASP_MISS
                    self.this_pmt.error = "GRASP_MISS"
                    self.logger.error(f'[Grasp] - GRASP_MISS')
                else:
                    self.this_pmt.ret = self.SUCCESS
                    self.this_pmt.error = "NONE"
                    self.logger.info(f'[Grasp] - Success When Grasping')
        
        self.logger.info(f'[Grasp] - Result - action: {self.this_pmt.action} ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        
        self.this_pmt.param = [self.dx,self.dy,self.R]
        self.update()
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        
        self.logger.flag(f'[Grasp] - Grasp End')
        end_time = time.time()
        self.logger.time(f'[Grasp] - Grasp Time: {end_time-start_time} s')

        print(f'========== Grasp Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def unlock(self,param):
        print(f'========== Unlocking ... ==========')
        
        self.logger.flag(f'[Unlock] - Unlock Start')
        start_time = time.time()

        self.this_pmt.action = "UNLOCK"
        self.this_pmt.id = self.UNLOCK
        self.this_pmt.param = [0,0,0]

        ## close gripper
        # print(f'Closing Gripper ...')
        self.logger.flag(f'[Unlock] - Gripper Closing Start')
        self.arm.control_gripper(self.cfg.unlock.gripper_value_before)
        time.sleep(2)
        self.logger.flag(f'[Unlock] - Gipper Value Before: {self.cfg.unlock.gripper_value_before}')
        self.logger.flag(f'[Unlock] - Gripper Closing End')

        ## Current Detection Begin
        if self.r_l == 'right':
            self.start_current_monitor_thread(thresholds_safety=self.unlock_thresholds,thresholds_event=self.unlock_stop_thresholds_right,if_event_stop=True)
        elif self.r_l == 'left':
            self.start_current_monitor_thread(thresholds_safety=self.unlock_thresholds,thresholds_event=self.unlock_stop_thresholds_left,if_event_stop=True)
        self.logger.info(f'[Unlock] - Current Detection Begin')

        ## unlock
        # print(f'Unlocking ...')
        self.logger.flag(f'[Unlock] - Moving Start')
        tag = self.arm.move_p(pos=self.p2_3d_base_xyzrxryrz,vel=self.cfg.unlock.unlock_v,if_p=True)
        time.sleep(2)
        self.logger.info(f'[Unlock] - Moving Result - tag: {tag} unlock vel: {self.cfg.unlock.unlock_v}')
        self.logger.flag(f'[Unlock] - Moving End')
        
        ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()
        self.logger.info(f'[Unlock] - Current Detection End')

        ## close gripper
        if not self.this_pmt.ret == self.SAFETY_ISSUE:
            # print(f'Closing Gripper ...')
            self.logger.flag(f'[Unlock] - Gripper Closing Start')
            self.arm.control_gripper(self.cfg.unlock.gripper_value_after)
            time.sleep(2)
            self.logger.flag(f'[Unlock] - Gipper Value After: {self.cfg.unlock.gripper_value_after}')
            self.logger.flag(f'[Unlock] - Gripper Closing End')
        
        ## clip
        text_prompt=["handle with gripper grasped firmly", "handle without gripper grasped firmly"]
        clip_result,clip_text,clip_probs = self.clip.clip_detection(rgb_img=self.rgb_img_path,text_prompt=text_prompt,if_p=False)
        clip_success = (clip_result == 0 and self.arm.get_gripper_grasp_return(if_p=False) == 2) or (clip_result == 1 and self.arm.get_gripper_grasp_return(if_p=False) != 2)
        self.logger.info(f'[Unlock] - CLIP Result - clip_success: {clip_success} text_prompt: {text_prompt} clip_result: {clip_result} clip_text: {clip_text} clip_probs: {clip_probs}')
        
        ## gemini
        gemini_result,gemini_text,gemini_probs =  self.gemini.gemini_detection(img_path=self.rgb_img_path,text_prompt=text_prompt,if_p=False)
        gemini_success = (gemini_result == 0 and self.arm.get_gripper_grasp_return(if_p=False) == 2) or (gemini_result == 1 and self.arm.get_gripper_grasp_return(if_p=False) != 2)
        self.logger.info(f'[Unlock] - gemini Result - gemini_success: {gemini_success} text_prompt: {text_prompt} gemini_result: {gemini_result} gemini_text: {gemini_text} gemini_probs: {gemini_probs}')

        ## update
        if self.this_pmt.ret == self.SAFETY_ISSUE:
            self.this_pmt.ret = self.UNLOCK_SAFETY
            self.this_pmt.error = "UNLOCK_SAFETY"
            self.logger.error(f'[Unlock] - UNLOCK_SAFETY')
        elif self.this_pmt.ret == self.EVENT_DETECTED:
            if self.arm.get_gripper_grasp_return(if_p=False) != 2:
                self.this_pmt.ret = self.UNLOCK_MISS
                self.this_pmt.error = "UNLOCK_MISS"
                self.logger.error(f'[Unlock] - UNLOCK_MISS')
            else:
                self.this_pmt.ret = self.SUCCESS
                self.this_pmt.error = "NONE"
                self.logger.info(f'[Unlock] - Success When Unlocking')
        elif self.this_pmt.ret == self.NO_ISSUE:
            if tag:
                self.this_pmt.ret = self.UNLOCK_IK_FAIL
                self.this_pmt.error = "UNLOCK_IK_FAIL"
                self.logger.error(f'[Unlock] - UNLOCK_IK_FAIL')
            elif self.arm.get_gripper_grasp_return(if_p=False) != 2:
                self.this_pmt.ret = self.UNLOCK_MISS
                self.this_pmt.error = "UNLOCK_MISS"
                self.logger.error(f'[Unlock] - UNLOCK_MISS')
            else:
                self.this_pmt.ret = self.SUCCESS
                self.this_pmt.error = "NONE"
                self.logger.info(f'[Unlock] - Success When Unlocking')
        
        unlock_T = self.action_T * np.sign(self.R)
        self.this_pmt.param = [unlock_T,0,0]
    
        self.logger.info(f'[Unlock] - unlock_T: {unlock_T}')
        self.logger.info(f'[Unlock] - Result - action: {self.this_pmt.action} ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')

        self.update()
        
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        self.logger.flag(f'[Unlock] - Unlock End')
        end_time = time.time()
        self.logger.time(f'[Unlock] - Unlock Time: {end_time-start_time} s')

        print(f'========== Unlock Done ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def rotate(self,param):
        print(f'========== Rotating ... ==========')

        self.logger.flag(f'[Rotate] - Rotate Start')
        start_time = time.time()

        self.this_pmt.action = "ROTATE"
        self.this_pmt.id = self.ROTATE
        self.this_pmt.param = [0,0,0]

        ## close gripper
        # print(f'Closing Gripper ...')
        self.logger.flag(f'[Rotate] - Gripper Closing Start')
        self.arm.control_gripper(self.cfg.rotate.gripper_value)
        time.sleep(1)
        self.logger.flag(f'[Rotate] - Gipper Value Before: {self.cfg.rotate.gripper_value}')
        self.logger.flag(f'[Rotate] - Gripper Closing End')

        ## Current Detection Begin
        if self.r_l == 'right':
            self.start_current_monitor_thread(thresholds_safety=self.rotate_thresholds,thresholds_event=self.rotate_stop_thresholds_right,if_event_stop=True)
        elif self.r_l == 'left':
            self.start_current_monitor_thread(thresholds_safety=self.rotate_thresholds,thresholds_event=self.rotate_stop_thresholds_left,if_event_stop=True)
        self.logger.info(f'[Rotate] - Current Detection Start')

        ## rotate
        # print(f'Rotating ...')
        self.logger.flag(f'[Rotate] - Moving Start')
        joint = self.arm.get_j()
        joint[6] += 75
        tag = self.arm.move_j(joint=joint,vel=self.cfg.rotate.rotate_v)
        time.sleep(2)
        self.logger.info(f'[Rotate] - Moving Result - tag: {tag} rotate vel: {self.cfg.rotate.rotate_v}')
        self.logger.flag(f'[Rotate] - Moving End')

        ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()
        self.logger.info(f'[Rotate] - Current Detection End')

        ## clip
        text_prompt=["handle with gripper grasped firmly", "handle without gripper grasped firmly"]
        clip_result,clip_text,clip_probs = self.clip.clip_detection(rgb_img=self.rgb_img_path,text_prompt=text_prompt,if_p=False)
        clip_success = (clip_result == 0 and self.arm.get_gripper_grasp_return(if_p=False) == 2) or (clip_result == 1 and self.arm.get_gripper_grasp_return(if_p=False) != 2)
        self.logger.info(f'[Rotate] - CLIP Result - clip_success: {clip_success} text_prompt: {text_prompt} clip_result: {clip_result} clip_text: {clip_text} clip_probs: {clip_probs}')
        
        ## gemini
        # gemini_result,gemini_text,gemini_probs =  self.gemini.gemini_detection(img_path=self.rgb_img_path,text_prompt=text_prompt,if_p=False)
        # gemini_success = (gemini_result == 0 and self.arm.get_gripper_grasp_return(if_p=False) == 2) or (gemini_result == 1 and self.arm.get_gripper_grasp_return(if_p=False) != 2)
        # self.logger.info(f'[Rotate] - gemini Result - gemini_success: {gemini_success} text_prompt: {text_prompt} gemini_result: {gemini_result} gemini_text: {gemini_text} gemini_probs: {gemini_probs}')

        ## update
        if self.this_pmt.ret == self.SAFETY_ISSUE:
            self.this_pmt.ret = self.ROTATE_SAFETY
            self.this_pmt.error = "ROTATE_SAFETY"
            self.logger.error(f'[Rotate] - ROTATE_SAFETY When Rotating')
        elif self.this_pmt.ret == self.NO_ISSUE or self.this_pmt.ret == self.EVENT_DETECTED:
            if tag != 0:
                self.this_pmt.ret = self.ROTATE_IK_FAIL
                self.this_pmt.error = "ROTATE_IK_FAIL"
                self.logger.error(f'[Rotate] - ROTATE_IK_FAIL When Rotating')
            elif self.arm.get_gripper_grasp_return(if_p=False) != 2:
                self.this_pmt.ret = self.ROTATE_MISS
                self.this_pmt.error = "ROTATE_MISS"
                self.logger.error(f'[Rotate] - ROTATE_MISS When Rotating')
            else:
                self.this_pmt.ret = self.SUCCESS
                self.this_pmt.error = "NONE"
                self.logger.info(f'[Rotate] - Success When Rotating')

        rotate_T = self.action_T
        self.this_pmt.param = [rotate_T,0,0]

        self.logger.info(f'[Rotate] - rotate_T: {rotate_T}')
        self.logger.info(f'[Rotate] - Result - action: {self.this_pmt.action} ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')

        self.update()
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')

        self.logger.flag(f'[Rotate] - Rotate End')
        end_time = time.time()
        self.logger.time(f'[Rotate] - Rotate Time: {end_time-start_time} s')

        print(f'========== Rotate Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def open_directly(self,param):
        print(f'========== Opening ... ==========')
        start_time = time.time()

        self.this_pmt.action = "OPEN"
        self.this_pmt.id = self.OPEN
        self.this_pmt.param = [0,0,0]

        if self.r_l == 'right':
            direction = -1
        elif self.r_l == 'left':
            direction = 1
        
        ## for push directly
        self.ps_pl = 1
        ## for pull directly
        self.ps_pl = 0
        ## for Exp3: open loop
        self.ps_pl = 1 if random.random() < 0.5 else -1
        
        # print(f'ps_pl: {ps_pl}')
        self.logger.info(f'[Open] - ps_pl: {ps_pl}')

        ## Current Detection Begin
        self.start_current_monitor_thread(thresholds_safety=self.open_thresholds,if_event_stop=False)
        self.logger.info(f'[Open] - Current Detection Start')

        ## close gripper
        # print(f'Close Gripper ...')
        self.logger.flag(f'[Open] - Gripper Closing Start')
        self.arm.control_gripper(self.cfg.open.gripper_value_pull)
        time.sleep(2)
        self.logger.info(f'[Open] - Gipper Value: {self.cfg.open.gripper_value_pull}')
        self.logger.flag(f'[Open] - Gripper Closing End')

        ## open
        # print(f'opening ...')
        self.logger.flag(f'[Open] - Open Start')
        self.base.move_open_door(self.cfg.open.open_T,self.ps_pl*self.cfg.open.linear_velocity,self.cfg.open.angular_velocity*direction)
        time.sleep(3)
        self.logger.info(f'[Open] - open_T: {self.cfg.open.open_T} linear_velocity: {self.cfg.open.linear_velocity} angular_velocity: {self.cfg.open.angular_velocity}')
        self.logger.flag(f'[Open] - Open End')

        ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()
        self.logger.info(f'[Open] - Current Detection End')

        if self.this_pmt.ret == self.SAFETY_ISSUE:
            self.this_pmt.ret = self.OPEN_SAFETY
            self.this_pmt.error = "OPEN_SAFETY"
            self.logger.error(f'[Open] - OPEN_SAFETY')
        elif self.this_pmt.ret == self.NO_ISSUE:
            self.this_pmt.ret = self.SUCCESS
            self.this_pmt.error = "NONE"
            self.logger.info(f'[Open] - Success When Opening')

        self.this_pmt.param = [self.ps_pl,self.cfg.open.open_T,0]

        self.update()
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')

        self.logger.info(f'[Open] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[Open] - Open End')
        end_time = time.time()
        self.logger.time(f'[Open] - Open Time: {end_time-start_time} s')
        
        print(f'========== Open Done ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def open(self,param):
        print(f'========== Opening ... ==========')
        
        self.logger.flag(f'[Open] - Open Start')
        start_time = time.time()

        self.this_pmt.action = "OPEN"
        self.this_pmt.id = self.OPEN
        self.this_pmt.param = [0,0,0]

        if self.r_l == 'right':
            self.open_pull_thresholds = self.open_pull_thresholds_right
            direction = -1
        elif self.r_l == 'left':
            self.open_pull_thresholds = self.open_pull_thresholds_left
            direction = 1
        self.logger.info(f'[Open] - Arm Choice: {self.r_l} Open Pull Threshoulds: {self.open_pull_thresholds}')

        if param:
            self.open_pull_thresholds[3][0] = param[0] # pull_j3_l
            self.cfg.open.explore_T = param[1]
            self.cfg.open.explore_vel = param[2]
            self.cfg.open.open_T = param[3]
            self.cfg.open.linear_velocity = param[4]

        self.ps_pl = 0

        ## Current Detection Begin
        self.start_current_monitor_thread(thresholds_safety=self.open_thresholds,thresholds_event=self.open_pull_thresholds,if_event_stop=False)
        self.logger.info(f'[Open] - Current Detection Start')

        ## close gripper
        # print(f'Close Gripper ...')
        self.logger.flag(f'[Open] - Gripper Closing Start')
        self.arm.control_gripper(self.cfg.open.gripper_value_pull)
        time.sleep(2)
        self.logger.info(f'[Open] - Gipper Value: {self.cfg.open.gripper_value_pull}')
        self.logger.flag(f'[Open] - Gripper Closing End')

        ## explore (pull)
        # print(f'exploring (pull)...')
        self.logger.flag(f'[Open] - Explore Start')
        self.base.move_T(-self.cfg.open.explore_T,self.cfg.open.explore_vel)
        time.sleep(1)
        self.logger.info(f'[Open] - explore_T: {self.cfg.open.explore_T} explore_vel: {self.cfg.open.explore_vel}')
        self.logger.flag(f'[Open] - Explore End')

        ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()
        self.logger.info(f'[Open] - Current Detection End')

        if self.this_pmt.ret != self.SAFETY_ISSUE:
            ## handle slip
            if self.arm.get_gripper_grasp_return(if_p=False) != 2:
                self.this_pmt.ret = self.OPEN_MISS
                self.this_pmt.error = "OPEN_MISS"
                self.logger.error(f'[Open] - OPEN_MISS When Exploring (Pull)')
            else:
                ## detect that it should be a push door
                if self.this_pmt.ret == self.EVENT_DETECTED:

                    self.logger.info(f'[Open] - Detect the door is a push door')

                    ## Current Detection Begin
                    self.start_current_monitor_thread(thresholds_safety=self.open_thresholds)
                    
                    ## open (push)
                    # print(f'opening (push)...')
                    self.logger.flag(f'[Open] - Push Start')
                    self.base.move_open_door(self.cfg.open.open_T,self.cfg.open.linear_velocity,self.cfg.open.angular_velocity*direction)
                    time.sleep(3)
                    self.logger.info(f'[Open] - open_T: {self.cfg.open.open_T} linear_velocity: {self.cfg.open.linear_velocity} angular_velocity: {self.cfg.open.angular_velocity}')
                    self.logger.flag(f'[Open] - Push End')
                    
                    ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
                    self.monitor_running = False
                    self.current_monitor_thread.join()
                    self.vis_current_data()

                    if self.this_pmt.ret == self.NO_ISSUE:
                        self.ps_pl = 1 # 'push'
                        # print(f'pushing successed ...')
                        self.this_pmt.ret = self.SUCCESS
                        self.this_pmt.error = "NONE"
                        self.logger.info(f'[Open] - Pushing Successed')

                    elif self.this_pmt.ret == self.SAFETY_ISSUE:
                        self.this_pmt.ret = self.OPEN_SAFETY
                        self.this_pmt.error = "OPEN_SAFETY"
                        self.logger.error(f'[Open] - OPEN_SAFETY When Pushing')
                    
                ## detect that it should be a pull door
                else:
                    self.logger.info(f'[Open] - Detect the door is a pull door')

                    ## Current Detection Begin
                    self.start_current_monitor_thread(thresholds_safety=self.open_thresholds)
                    
                    ## open (pull)
                    # print(f'opening (pull)...')
                    self.logger.flag(f'[Open] - Pull Start')
                    self.base.move_open_door(self.cfg.open.open_T,-self.cfg.open.linear_velocity,-self.cfg.open.angular_velocity*direction)
                    time.sleep(3)
                    self.logger.info(f'[Open] - open_T: {self.cfg.open.open_T} linear_velocity: {self.cfg.open.linear_velocity} angular_velocity: {self.cfg.open.angular_velocity}')
                    self.logger.flag(f'[Open] - Pull End')

                    ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
                    self.monitor_running = False
                    self.current_monitor_thread.join()
                    self.vis_current_data()

                    if self.this_pmt.ret == self.NO_ISSUE:
                        self.ps_pl = -1 # 'pull'
                        # print(f'pushing successed ...')
                        self.this_pmt.ret = self.SUCCESS
                        self.this_pmt.error = "NONE"
                        self.logger.info(f'[Open] - Pulling Successed')

                    elif self.this_pmt.ret == self.SAFETY_ISSUE:
                        self.this_pmt.ret = self.OPEN_SAFETY
                        self.this_pmt.error = "OPEN_SAFETY"
                        self.logger.error(f'[Open] - OPEN_SAFETY When Pulling')

        else:
            self.this_pmt.ret = self.OPEN_SAFETY
            self.this_pmt.error = "OPEN_SAFETY"
            self.logger.error(f'[Open] - OPEN_SAFETY When Exploring')

        ## clip
        text_prompt=["handle with gripper grasped firmly", "handle without gripper grasped firmly"]
        clip_result,clip_text,clip_probs = self.clip.clip_detection(rgb_img=self.rgb_img_path,text_prompt=text_prompt,if_p=False)
        clip_success = (clip_result == 0 and self.arm.get_gripper_grasp_return(if_p=False) == 2) or (clip_result == 1 and self.arm.get_gripper_grasp_return(if_p=False) != 2)
        self.logger.info(f'[Open] - CLIP Result - clip_success: {clip_success} text_prompt: {text_prompt} clip_result: {clip_result} clip_text: {clip_text} clip_probs: {clip_probs}')
        
        # ## gemini
        # gemini_result,gemini_text,gemini_probs =  self.gemini.gemini_detection(img_path=self.rgb_img_path,text_prompt=text_prompt,if_p=False)
        # gemini_success = (gemini_result == 0 and self.arm.get_gripper_grasp_return(if_p=False) == 2) or (gemini_result == 1 and self.arm.get_gripper_grasp_return(if_p=False) != 2)
        # self.logger.info(f'[Open] - gemini Result - gemini_success: {gemini_success} text_prompt: {text_prompt} gemini_result: {gemini_result} gemini_text: {gemini_text} gemini_probs: {gemini_probs}')


        self.this_pmt.param = [self.ps_pl,self.cfg.open.open_T,0]

        self.update()
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')

        self.logger.info(f'[Open] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[Open] - Open End')
        end_time = time.time()
        self.logger.time(f'[Open] - Open Time: {end_time-start_time} s')
        
        print(f'========== Open Done ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def swing(self,param): ## ONLY FOR PULL DOOR
        print(f'========== Swing ... ==========')
        
        self.logger.flag(f'[Swing] - Swing Start')
        start_time = time.time()

        self.this_pmt.action = "SWING"
        self.this_pmt.id = self.SWING
        self.this_pmt.param = [0,0,0]

        ## right or left
        if self.r_l == 'right':
            swing_pos1 = self.cfg.swing.pos1_right
            swing_pos2 = self.cfg.swing.pos2_right
        elif self.r_l == 'left':
            swing_pos1 = self.cfg.swing.pos1_left
            swing_pos2 = self.cfg.swing.pos2_left

        ## Current Detection Begin
        self.start_current_monitor_thread(thresholds_safety=self.swing_thresholds)
        self.logger.info(f'[Swing] - Current Detection Start')

        ## 1. for not rebounce device
        # ## arm2 move
        # self.logger.info(f'[Swing] - Arm2 Moving Start Pos1')
        # self.arm2.move_p(pos=swing_pos1,vel=self.cfg.swing.swing_v,if_p=True,block=True)
        # self.logger.flag(f'[Swing] - swing_pos1: {swing_pos1} swing_v: {self.cfg.swing.swing_v}')
        # self.logger.info(f'[Swing] - Arm2 Moving End Pos1')

        # ## close gripper
        # # print(f'Close Gripper ...')
        # self.logger.flag(f'[Swing] - Gripper Closing Start')
        # self.arm.control_gripper(self.cfg.swing.gripper_value)
        # time.sleep(2)
        # self.logger.info(f'[Swing] - Gipper Value: {self.cfg.swing.gripper_value}')
        # self.logger.flag(f'[Swing] - Gripper Closing End')

        # ## back to home
        # # print(f'Back to home ...')
        # self.logger.info(f'[Swing] - Arm1 Moving Back to Home Start')
        # self.arm.move_j(joint=self.p1_joint,vel=self.cfg.swing.swing_v,if_p=True,block=True)
        # self.arm.move_j(joint=self.arm.middle_state,vel=self.cfg.swing.swing_v,if_p=True,block=True)
        # self.arm.go_home(vel=self.cfg.swing.swing_v,block=True)
        # self.logger.info(f'[Swing] - Arm1 Moving Back to Home End')

        # ## move to open completely
        # # print(f'Moving ...')
        # self.logger.info(f'[Swing] - Arm2 Moving Start Pos2')
        # self.base.move_T(T=self.cfg.swing.base_move_T,linear_velocity=self.cfg.swing.base_move_v)
        # time.sleep(2)
        # self.base.rotate_T(T=self.cfg.swing.base_rotate_T,angular_velocity=self.cfg.swing.base_rotate_v)
        # self.arm2.move_p(pos=swing_pos2,vel=self.cfg.swing.swing_v,if_p=True,block=True)
        # self.arm2.go_home(vel=self.cfg.swing.swing_v,block=True)
        # self.logger.flag(f'[Swing] - swing_pos2: {swing_pos2} swing_v: {self.cfg.swing.swing_v}')
        # self.logger.info(f'[Swing] - Arm2 Moving End Pos2')

        
        # ## 2. for rebounce device (only arm_right can do)
        self.arm2.move_p(pos=[0.637474000453949, -0.507669985294342, -0.173102006316185, -1.4700000286102295, -1.0570000410079956, -2.0920000076293945],if_p=True,vel=self.cfg.swing.swing_v)
        self.arm2.move_p(pos=[0.6744419932365417, -0.25297001004219055, -0.2421559989452362, -1.6160000562667847, 0.453000009059906, -1.4359999895095825],if_p=True,vel=self.cfg.swing.swing_v)
        # # self.arm.control_gripper(open_value=1000,block=False)
        # # self.arm.go_home(vel=self.cfg.swing.swing_v)
        # # self.base.move_T(T=self.cfg.swing.base_move_T,linear_velocity=self.cfg.swing.base_move_v)
       
        # ## method 1:
        # # self.arm2.move_p(pos=[0.6482849717140198, -0.04628700017929077, -0.22619999945163727, -1.50600004196167, 0.43799999356269836, -1.2259999513626099],if_p=True,vel=self.cfg.swing.swing_v)
        # # self.arm.control_gripper(open_value=1000,block=False)
        # # self.arm.move_j(joint=[3.6589999198913574, 78.0780029296875, -163.43499755859375, 126.53700256347656, -85.34100341796875, -70.31999969482422, 165.06100463867188],vel=self.cfg.swing.swing_v,if_p=True)

        # ## method 2:
        self.arm.control_gripper(open_value=1000,block=False)
        self.arm.move_j(joint=[3.6589999198913574, 78.0780029296875, -163.43499755859375, 126.53700256347656, -85.34100341796875, -70.31999969482422, 165.06100463867188],vel=self.cfg.swing.swing_v,if_p=True,block=False)
        time.sleep(0.1)
        self.arm2.move_p(pos=[0.6482849717140198, -0.04628700017929077, -0.22619999945163727, -1.50600004196167, 0.43799999356269836, -1.2259999513626099],if_p=True,vel=self.cfg.swing.swing_v,block=True)

        self.arm.go_home(vel=self.cfg.swing.swing_v)
        self.base.move_T(T=self.cfg.swing.base_move_T,linear_velocity=self.cfg.swing.base_move_v)
        
        self.arm.move_p(pos=[0.2763560116291046, -0.6074569821357727, 0.30012598633766174, -1.2920000553131104, 0.5099999904632568, 3.1040000915527344],if_p=True,vel=self.cfg.swing.swing_v)
        self.arm.move_p(pos=[0.198745995759964, -0.3270829916000366, 0.8150449991226196, -0.5600000023841858, 0.460999995470047, -2.861999988555908],if_p=True,vel=self.cfg.swing.swing_v)
        self.arm2.move_p(pos=[0.637474000453949, -0.507669985294342, -0.173102006316185, -1.4700000286102295, -1.0570000410079956, -2.0920000076293945],if_p=True,vel=self.cfg.swing.swing_v)
        self.arm2.go_home(vel=self.cfg.swing.swing_v)

        ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()
        self.logger.info(f'[Swing] - Current Detection End')

        ## update
        if self.this_pmt.ret == self.NO_ISSUE:
            self.this_pmt.ret = self.SUCCESS
            self.this_pmt.error = "NONE"
            self.logger.info(f'[Swing] - Success When Swing')
        elif self.this_pmt.ret == self.SAFETY_ISSUE:
            self.this_pmt.ret = self.SWING_SAFETY
            self.this_pmt.error = "SWING_SAFETY"
            self.logger.error(f'[Swing] - SWING_SAFETY')

        self.update()
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')

        self.logger.info(f'[Swing] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[Swing] -  Swing End')
        end_time = time.time()
        self.logger.time(f'[Swing] - Swing Time: {end_time-start_time} s')

        print(f'========== Swing Done ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def teleoperation(self):
        print(f'========== Teleoperation ... ==========')

        self.logger.flag(f'[Tele] - Tele Start')
        start_time = time.time()

        while True:
            try: 
                char = getch(if_p=True)
                if char in ['w','a','s','d','H','P','K','M']:
                    self.base.move_char(char,linear_velocity=self.cfg.tele.base_linear_velocity,angular_velocity=self.cfg.tele.base_angular_velocity)
                elif char == '0':
                    self.arm.control_gripper(self.cfg.back.gripper_value)
                elif char == '1':
                    self.arm.move_p(pos=self.p1_3d_base_xyzrxryrz,vel=self.cfg.tele.arm_v,if_p=True,block=False)
                elif char == '2':
                    self.arm.go_home(vel=self.cfg.tele.arm_v,block=False)
                elif char == 'q':
                    break
                time.sleep(0.1)
            except KeyboardInterrupt:
                break

        self.this_pmt.action = "TELEOPERATION"
        self.this_pmt.id = self.TELEOPERATION
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "None"

        self.update()
        
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        self.logger.info(f'[Tele] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[Tele] - Tele End')
        end_time = time.time()
        self.logger.time(f'[Tele] - Tele Time: {end_time-start_time} s')

        print(f'========== Finish Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def telearml(self):
        print(f'========== TeleArmL ... ==========')

        self.logger.flag(f'[TeleArmL] - TeleArmL Start')
        start_time = time.time()
        self.arm_l.run()

        self.this_pmt.action = "TELEARML"
        self.this_pmt.id = self.TELEARML
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "None"

        self.update()
        
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        self.logger.info(f'[TeleArmL] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[TeleArmL] - TeleArmL End')
        end_time = time.time()
        self.logger.time(f'[TeleArmL] - TeleArmL Time: {end_time-start_time} s')

        print(f'========== TeleArmL Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def telearmr(self):
        print(f'========== TeleArmR ... ==========')

        self.logger.flag(f'[TeleArmR] - TeleArmR Start')
        start_time = time.time()
        self.arm_r.run()

        self.this_pmt.action = "TELEARMR"
        self.this_pmt.id = self.TELEARMR
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "None"

        self.update()
        
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        self.logger.info(f'[TeleArmR] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[TeleArmR] - TeleArmR End')
        end_time = time.time()
        self.logger.time(f'[TeleArmR] - TeleArmR Time: {end_time-start_time} s')

        print(f'========== TeleArmR Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def home(self):
        print(f'========== Going Home ... ==========')

        self.logger.flag(f'[Home] - Home Start')
        start_time = time.time()

        self.arm.control_gripper(self.cfg.home.gripper_value)
        time.sleep(2)
        self.base.move_T(-self.cfg.home.move_T)
        time.sleep(1)
        self.arm.go_home()
        self.base.move_T(self.cfg.home.move_T)
        time.sleep(2)

        self.this_pmt.action = "HOME"
        self.this_pmt.id = self.HOME
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "None"

        self.update()
        
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        self.logger.info(f'[Home] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[Home] - Home End')
        end_time = time.time()
        self.logger.time(f'[Home] - Home Time: {end_time-start_time} s')

        print(f'========== Home Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def back(self):
        ## back to grasping state
        print(f'========== Backing to p1 ... ==========')
        
        self.logger.flag(f'[Back] - Back Start')
        start_time = time.time()
        
        self.arm.control_gripper(self.cfg.back.gripper_value)
        time.sleep(2)
        # tag = self.arm.move_p(pos=self.p1_3d_base_xyzrxryrz,if_p=True)
        tag = self.arm.move_j(joint=self.p1_joint,if_p=True)

        self.this_pmt.action = "BACK"
        self.this_pmt.id = self.BACK
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "BACK"
        
        self.update()

        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        self.logger.info(f'[Back] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[Back] - Back End')
        end_time = time.time()
        self.logger.time(f'[Back] - Back Time: {end_time-start_time} s')

        print(f'========== Back Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def finish(self):
        print(f'========== Finishing ... ==========')
        
        self.logger.flag(f'[Finish] - Finish Start')
        start_time = time.time()

        self.arm.control_gripper(self.cfg.finish.gripper_value)
        time.sleep(2)
        self.base.move_T(-self.cfg.finish.move_T)
        time.sleep(1)
        self.arm.go_home()
        self.base.move_location([self.base.start_x,self.base.start_y,self.base.start_theta])
        time.sleep(1)
        self.base.move_T(self.cfg.finish.move_T)
        time.sleep(1)
        
        self.this_pmt.action = "FINISH"
        self.this_pmt.id = self.FINISH
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "FINISH"

        self.update()
        self.save_primitives()
        
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        self.logger.info(f'[Finish] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[Finish] - Finish End')
        end_time = time.time()
        self.logger.time(f'[Finish] - Finish Time: {end_time-start_time} s')

        print(f'========== Finish Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def clear(self):
        print(f'========== Clearing ... ==========')
        
        self.logger.info(f'[Clear] - Clear Start')
        start_time = time.time()

        self.action_num = 0

        self.last_pmt = _Primitive(action="START",id=self.START,ret=1,param=[0,0,0],error="START")
        self.this_pmt = _Primitive()

        self.primitives = {0:self.last_pmt.to_list()}
        
        self.current_max = [0]*7
        self.current_min = [0]*7
        
        if os.path.exists(self.tjt_dir):
             shutil.rmtree(self.tjt_dir)
        os.makedirs(self.tjt_dir)

        ret = 1
        error = "CLEAR"
        
        # print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        self.logger.info(f'[Clear] - action: {self.this_pmt.action} ret: {self.this_pmt.ret} error: {self.this_pmt.error}')

        self.logger.flag(f'[Clear] - Clear End')
        end_time = time.time()
        self.logger.time(f'[Clear] - Clear Time: {end_time-start_time} s')

        print(f'========== Clear Done... ==========')
        return ret,error
    
    @time_it
    def hl_VLM(self):
        img_path = f'{self.tjt_dir}/temp.png'
        last_pmt_action = self.last_pmt.action
        last_pmt_error = self.last_pmt.error
        hl_prompt = gen_hl_prompt(last_pmt_action,last_pmt_error)
        self.logger.info(f'[Gemini] - last_pmt_action: {last_pmt_action} last_pmt_error: {last_pmt_error}')
        response = self.gemini.text_img_to_text(text=hl_prompt,img_path=img_path,if_p=False)
        self.logger.info(f'[Gemini] - hl_VLM - response: {response}')
        return response
    
    @time_it
    def ll_VLM(self):
        img_path = f'{self.tjt_dir}/temp.png'
        # response = self.gemini.text_img_to_text(text=ll_prompt,img_path=img_path,if_p=False)

        example_param1 = [-42.79, 9.61, 94.04]
        example_param2 = [-45.11, 19.86, 2.28]
        example_img1 = f'{self.root_dir}/example_img/ll_example_img1.png'
        example_img2 = f'{self.root_dir}/example_img/ll_example_img2.png'
        ll_prompt = gen_ll_prompt(example_param1,example_param2)
        response = self.gemini.text_imgs_to_text(ll_prompt,img_paths=[example_img1,example_img2,img_path],if_p=False)
        self.logger.info(f'[Gemini] - ll_VLM - response: {response}')
        return response

    def do_primitive(self,_id,_param=None):
        primitive_type = self.action2num(_id)
        ## capture
        if primitive_type == -1:
            raise ValueError("Input error: Invalid primitive type value")
        elif primitive_type == self.CAPTURE:
            ret,error = 1,None
            self.capture(if_d=False,vis=False,if_update=False)
            return ret,error
        elif primitive_type == self.HLVLM or primitive_type == self.LLVLM:
            self.capture(if_d=False,vis=False,if_update=False)
        elif primitive_type == self.GRASP or primitive_type == self.PREMOVE:
            self.capture(if_d=True,vis=True,if_update=True)
        else:
            self.capture(if_d=False,vis=False,if_update=True)
    
        if _param:
            param = _param
        else:
            param = None

        ## do action
        if primitive_type == self.PREMOVE:
            ret,error = self.premove(param)
        elif primitive_type == self.GRASP:
            ret,error = self.grasp(param)
        elif primitive_type == self.ROTATE:
            ret,error = self.rotate(param)
        elif primitive_type == self.UNLOCK:
            ret,error = self.unlock(param)
        elif primitive_type == self.OPEN:
            ret,error = self.open(param)
        elif primitive_type == self.SWING:
            ret,error = self.swing(param)
        elif primitive_type == self.HOME:
            ret,error = self.home()
        elif primitive_type == self.FINISH:
            ret,error = self.finish()
        elif primitive_type == self.BACK:
            ret,error = self.back()
        elif primitive_type == self.CLEAR:
            ret,error = self.clear()
        elif primitive_type == self.TELEOPERATION:
            ret,error = self.teleoperation()
        elif primitive_type == self.TELEARML:
            ret,error = self.telearml()
        elif primitive_type == self.TELEARMR:
            ret,error = self.telearmr()
        elif primitive_type == self.HLVLM:
            response = self.hl_VLM()
            ret,error = 1,None
        elif primitive_type == self.LLVLM:
            response = self.ll_VLM()
            ret,error = 1,None
        self.capture(if_d=False,vis=False,if_update=False)
        
        return ret,error

    def run(self):
        num = 0
        while True:
            num += 1
            print('***************************************************************')
            user_input = input(f"[Please Input Primitive_{num}]: ")
            if user_input.lower() == 'q':
                break
            try:
                action_id, *param = [x.strip() for x in user_input.split(',')]
                param = [float(x) for x in param]
                if param:
                    ret, error = self.do_primitive(action_id, param)
                else:
                    ret, error = self.do_primitive(action_id)
                if error == 'FINISH':   
                    break
            except Exception as e:
                print(f"ERROR TYPE: {type(e)}: {e}")
                print("Please re-input!")
            print('***************************************************************\n\n')

    def open_loop(self):
        # ret, error = self.do_primitive(_id='premove')
        ret,error = self.do_primitive(_id='grasp')
        if self.type == 'lever':
            ret,error = self.do_primitive(_id='unlock')
        elif self.type == 'knob':
            ret,error = self.do_primitive(_id='rotate')
        ret, error = self.do_primitive(_id='open')
        ret, error = self.do_primitive(_id='tele')

    def close_loop_SM_GUM(self):
        state = 2
        while True:
            if state == 1:
                ret,error = self.do_primitive(_id='premove')
                if ret != 1:
                    state = 2
                else:
                    state = 2
            elif state == 2:
                ret,error = self.do_primitive(_id='grasp')
                if ret != 1:
                    if error == 'GRASP_IK_FAIL':
                        self.do_primitive(_id='home')
                        state = 1
                    elif error == 'GRASP_MISS':
                        self.do_primitive(_id='home')
                        state = 1
                    elif error == 'GRASP_SAFETY ':
                        self.do_primitive(_id='home')
                        state = 2
                else:
                    state = 3
            elif state == 3:
                if self.type == 'lever':
                    ret,error = self.do_primitive(_id='unlock')
                    if ret!= 1:
                        if error == 'UNLOCK_IK_FAIL':
                            self.do_primitive(_id='home')
                            state = 2
                        elif error == 'UNLOCK_MISS':
                            self.do_primitive(_id='back')
                            state = 3
                        elif error == 'UNLOCK_SAFETY':
                            self.do_primitive(_id='home')
                            state = 2
                    else:
                        state = 4
                elif self.type == 'knob':
                    ret,error = self.do_primitive(_id='rotate')
                    if ret!= 1:
                        if error == 'ROTATE_IK_FAIL':
                            self.do_primitive(_id='home')
                            state = 2
                        elif error == 'ROTATE_MISS':
                            self.do_primitive(_id='back')
                            state = 3
                        elif error == 'ROTATE_SAFETY':
                            self.do_primitive(_id='home')
                            state = 2
                    else:
                        state = 4
                else:
                    state = 4
            elif state == 4:
                ret,error = self.do_primitive(_id='open')
                if ret!= 1:
                    if error == 'OPEN_MISS':
                        self.do_primitive(_id='home')
                        state = 1
                    elif error == 'OPEN_SAFETY':
                        self.do_primitive(_id='home')
                        state = 2
                else:
                    state = 5
            elif state == 5:
                ret,error = self.do_primitive(_id='tele')
                break
# TODO

if __name__ == '__main__':
    primitive = Primitive(root_dir='./',tjt_num=2)
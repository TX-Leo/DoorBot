'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-04-30 16:43:14
Version: v1
File: 
Brief: 
'''
import numpy as np
import cv2
import os
import pyrealsense2 as rs
import sys
sys.path.append('../..')
from arm import Arm

def data_collection(arm,data_folder=None):
    def callback(frame):
        scaling_factor = 1.0
        global count
        cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        cv2.imshow("Capture_Video", cv_img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('s'):
            pose = arm.get_p()
            print(f'pose_{count}: {pose}')
            with open(f'{data_folder}\poses.txt', 'a+') as f:
                pose = [str(i) for i in pose]
                new_line = f'{",".join(pose)}\n'
                f.write(new_line)
            cv2.imwrite(f"{data_folder}\\{str(count)}.jpg" , cv_img)
            count += 1
        else:
            pass

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    global count
    count = 1
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            callback(color_image)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    data_folder = r'./data/left_arm_3/'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    arm = Arm.init_from_yaml(root_dir='../../',cfg_path='cfg/cfg_arm_left.yaml')

    data_collection(arm,data_folder)

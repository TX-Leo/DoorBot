'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-20 15:34:02
Version: v1
File: 
Brief: 
'''
import os
import cv2
import numpy as np
import logging

from save_poses_handineye import poses_main_handineye
from save_poses_handtoeye import poses_main_handtoeye

def handeye_calibration(data_folder,save_path="cam2base_H.csv",grid_num=(10,7),cell_width=2.18,pattern='hand_to_eye'):
    ## pictures num > 15
    matching_pictures = sorted([folder for folder in os.listdir(data_folder) if folder.endswith('.jpg') and folder.split(".")[0].isdigit()],key=lambda x:int(x.split('.')[0]))
    print(matching_pictures)
    if len(matching_pictures) >= 15:
        length,width = grid_num
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        objp = np.zeros((length * width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:length, 0:width].T.reshape(-1, 2)
        objp = cell_width / 100 * objp
        obj_points = []  # 3D points
        img_points = []  # 2D points
        for picture in matching_pictures:
            image = f"{data_folder}/{picture}"
            if os.path.exists(image):
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, (length, width), None)
                if ret:
                    obj_points.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),criteria)
                    if [corners2]:
                        img_points.append(corners2)
                    else:
                        img_points.append(corners)
        N = len(img_points)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        print(f"intrinsic matrix:\n{mtx}")
        print(f"distortion cofficients:\n{dist}" ) #(k_1,k_2,p_1,p_2,k_3)
        if pattern == 'hand_in_eye':
            poses_main_handineye(data_folder,f'{data_folder}/poses.txt')
        elif pattern == 'hand_to_eye':
            poses_main_handtoeye(data_folder,f'{data_folder}/poses.txt')
        else:
            print(f'[Error]: please input the correct pattern')
        tool_pose = np.loadtxt(f'{data_folder}/RobotToolPose.csv', delimiter=',')
        R_tool = []
        t_tool = []
        for i in range(int(N)):
            R_tool.append(tool_pose[0:3, 4 * i:4 * i + 3])
            t_tool.append(tool_pose[0:3, 4 * i + 3])

        # print('TSAI')
        # R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_TSAI)
        # print(R)
        # print(t)
        # print('HORAUD')
        # R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_HORAUD)
        # print(R)
        # print(t)
        # print('DANIILIDIS')
        # R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
        # print(R)
        # print(t)
        # print('ANDREFF')
        # R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_ANDREFF)
        # print(R)
        # print(t)

        ## Usually PARK Method is the best
        R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_PARK)
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t.flatten()
        print(f'cam2base_H (PARK Method):\n {H}')
        if save_path:
            np.savetxt(save_path, H, delimiter=",") 
    else:
        print("The number of pictures is smaller than 15, please collect more data!")

if __name__ == "__main__":
    data_folder = r'./data/left_arm/'
    save_path = f'{data_folder}/cam2base_H.csv'
    grid_num=(10,7)
    cell_width = 2.18182 # 24.0/11
    pattern = 'hand_to_eye' # or hand_in_eye
    handeye_calibration(data_folder,save_path,grid_num,cell_width,pattern)
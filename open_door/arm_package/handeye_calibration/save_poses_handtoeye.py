'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-19 02:02:18
Version: v1
File: 
Brief: 
'''
import csv
import numpy as np

def poses_main_handtoeye(dirpath,tag):
    with open(tag, "r",encoding="utf-8") as f:
        lines = f.readlines()
    lines = [float(i)  for line in lines for i in line.split(',')]
    matrices = []
    for i in range(0,len(lines),6):
        matrices.append(inverse_transformation_matrix(pose_to_homogeneous_matrix(lines[i:i+6])))
    save_matrices_to_csv(matrices, f'{dirpath}/RobotToolPose.csv')

def euler_angles_to_rotation_matrix(rx, ry, rz):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H

def inverse_transformation_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)
    T_inv = np.identity(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def save_matrices_to_csv(matrices, file_name):
    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))
    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)

if __name__ == "__main__":
    tag = ''
    poses2_main(tag)
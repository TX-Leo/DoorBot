import csv
import numpy as np

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

def poses_main_handineye(dirpath,filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [float(i) for line in lines for i in line.split(',')]
    matrices = []
    for i in range(0,len(lines),6):
        matrices.append(pose_to_homogeneous_matrix(lines[i:i+6]))
    save_matrices_to_csv(matrices, f'{dirpath}\RobotToolPose.csv')

def pose_convert(end_effector_pose):
    from scipy.spatial.transform import Rotation as R
    position = end_effector_pose[:3]
    orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
    print(f'orientation:{orientation}')
    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position
    return T_base_to_end_effector

if __name__ == "__main__":
    pose = [-0.079926997423172, 0.0061039999127388, 0.6547899842262268, 0.06199999898672104, 0.014000000432133675,
            -3.065000057220459]
    pose1 = [0,0,1,0,0,0]
    print(pose_convert(pose))
    print(pose_to_homogeneous_matrix(pose1))

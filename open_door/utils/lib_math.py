import numpy as np
import transforms3d as tfs
import cv2

def EulerAngle_to_R(rpy,rad=False):
    rpy = np.array(rpy)
    r,p,y = rpy
    if rad:
        R = tfs.euler.euler2mat(r, p, y, axes='sxyz')
    else:
        R = tfs.euler.euler2mat(np.radians(r), np.radians(p), np.radians(y), axes='sxyz')
    return R

def R_to_EulerAngle(R,rad=False):
    R = np.array(R)
    if rad:
        rpy = tfs.euler.mat2euler(R, 'sxyz')
    else:
        rpy = np.degrees(tfs.euler.mat2euler(R, 'sxyz'))
    return rpy

def Rvec_to_R(rvec):
    rvec = np.array(rvec)
    R, _ = cv2.Rodrigues(rvec)
    return R

def R_to_Rvec(R):
    R = np.array(R)
    Rvec, _ = cv2.Rodrigues(R)
    return Rvec

def xyz_to_t(xyz):
    xyz = np.array(xyz)
    t = xyz.reshape(3, 1)
    return t

def t_to_xyz(t):
    t = np.array(t)
    xyz = np.squeeze(t.reshape(1,3))
    return xyz

def xyz_rpy_to_xyzrpy(xyz,rpy):
    xyz = np.array(xyz)
    rpy = np.array(rpy)
    xyzrpy = np.concatenate((xyz, rpy), axis=0).flatten()
    return xyzrpy

def Rt_to_H(R,t):
    R = np.array(R)
    t = np.array(t)
    H = tfs.affines.compose(np.squeeze(t), R, [1, 1, 1])
    return H

def H_to_Rt(H):
    H = np.array(H)
    R = H[0:3, 0:3]
    t = H[0:3, 3]
    return R,t

def xyzrpy_to_H(xyzrpy,rad=False):
    xyzrpy = np.array(xyzrpy)
    R = EulerAngle_to_R(xyzrpy[3:],rad)
    t = xyz_to_t(xyzrpy[:3])
    H = Rt_to_H(R,t)
    return H

def H_to_xyzrpy(H,rad=False):
    H = np.array(H)
    R,t = H_to_Rt(H)
    xyz = t_to_xyz(t)
    rpy = R_to_EulerAngle(R,rad)
    xyzrpy = xyz_rpy_to_xyzrpy(xyz,rpy)
    return xyzrpy

def rotate_R(R, delta_r=0, delta_p=0, delta_y=0, rotate_order='zyx',rad=False):
    R = np.array(R)
    if not rad:
        delta_r = np.radians(delta_r)
        delta_p = np.radians(delta_p)
        delta_y = np.radians(delta_y)

    R_r = np.array([[1, 0, 0],
                     [0, np.cos(delta_r), -np.sin(delta_r)],
                     [0, np.sin(delta_r), np.cos(delta_r)]])
    R_p = np.array([[np.cos(delta_p), 0, np.sin(delta_p)],
                     [0, 1, 0],
                     [-np.sin(delta_p), 0, np.cos(delta_p)]])
    R_y = np.array([[np.cos(delta_y), -np.sin(delta_y), 0],
                     [np.sin(delta_y), np.cos(delta_y), 0],
                     [0, 0, 1]])

    s = f'np.dot(np.dot(np.dot(R, R_r{rotate_order[0]}),R_r{rotate_order[1]}),R_r{rotate_order[2]})'
    new_R = eval(s)
    return new_R

def rotate_xyzrpy(xyzrpy,delta_r=0,delta_p=0,delta_y=0,rotate_order='zyx',rad=False):
    xyzrpy = np.array(xyzrpy)
    R = EulerAngle_to_R(xyzrpy[3:]) # R = Rvec_to_R(np.array(xyzrpy[3:]))
    new_R = rotate_R(R, delta_r,delta_p,delta_y,rotate_order,rad)
    new_rpy = R_to_EulerAngle(new_R)
    new_xyzrpy = xyz_rpy_to_xyzrpy(np.array(xyzrpy[:3]), new_rpy).tolist()
    return new_xyzrpy

def normal2rxryrz(normal,if_p=False):
        from scipy.spatial.transform import Rotation as R
        original_normal = np.array(normal)
        normal = original_normal * -1
        z_axis = normal / np.linalg.norm(normal)
        initial_x_axis = np.array([1, 0, 0])
        x_axis = initial_x_axis - np.dot(initial_x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')
        rx,ry,rz = euler_angles
        if if_p:
            print(f'original_normal:\n{original_normal}')
            print(f'normal:\n{normal}')
            print(f'z_axis:\n{z_axis}')
            print(f'initial_x_axis:\n{initial_x_axis}')
            print(f'x_axis:\n{x_axis}')
            print(f'y_axis:\n{y_axis}')
            print(f'rotation_matrix:\n{rotation_matrix}')
            print(f'euler_angles:\n{euler_angles}')
            print(f'rx:{rx} ry:{ry} rz:{rz}')
        return rx,ry,rz

def test1():
    R = [[-0.06472430155853726,-0.9973381232093603,-0.03357726583553946,],[-0.9973485867817671,0.06353005346138718,0.03549265771403007],[-0.033265015138603936,0.03578547611006781,-0.9988057060647002]]
    R= [[-0.99978544 , 0.01618313,  0.01292909],[ 0.01924667 , 0.95650893  ,0.29106747],[-0.00765641  ,0.29125386 ,-0.95661516]]
    EulerAngle = R_to_EulerAngle(np.array(R))
    for i in EulerAngle:
        print(np.radians(i))
    print(EulerAngle)

def test2():
    cam2baselink_H = xyzrpy_to_H(np.array([0.04259999,0.02724008,1.234697,-111.8845794,1.55444518,-92.16392625]))
    print(f'cam2baselink_H:\n{cam2baselink_H}')
    rlink12baselink_H = xyzrpy_to_H(np.array([-0.0096088,-0.17189,1.1246,90,0,-64.9724037791713]))
    print(f'rlink12baselink_H:\n{rlink12baselink_H}')
    cam2rlink1_H = cam2baselink_H @ np.linalg.inv(rlink12baselink_H) # cam2baselink_H @ rlink12baselink_H
    print(f'cam2rlink1_H:\n{cam2rlink1_H}')

def test3():
    rubik_center = [0.46196093592231313,0.16167266059444962,-0.516,1]
    r_cam2base_H = np.array([[0.01155362,0.84610748,0.53288708,-0.1165359],[-0.4203419,-0.47943922,0.77035753,0.14733703],[0.90729224,-0.23289519,0.35011516,-0.02538692],[0,0,0,1]])
    print(np.dot(r_cam2base_H, rubik_center)) # 0.3005636  0.27314776 0.53675329 1.

def test4():
    R = [[-0.3220086,-0.19165227,0.9271353],[-0.91081784,0.32990057,-0.24814607],[-0.2583047,-0.92435655,-0.28079113]]
    rpy = R_to_EulerAngle(R,rad=True)
    print(rpy)

def test5():
    # target2cam_rpy = [0.32611863105745464, -0.09393242756730547, -0.03171004262462371]
    # target2cam_rpy = [0.3624729993914516,-0.0035345173383167783, -0.001340387386237124]
    target2cam_rpy = [0.3631291403893837, -0.008841798828976621, -0.003359646326433152]
    target2base_rpy = [-1.0709999799728394, 0.4970000088214874, -3.072000026702881]

    target2cam_R = EulerAngle_to_R(target2cam_rpy,rad=True)
    target2base_R = EulerAngle_to_R(target2base_rpy,rad=True)

    cam2base_R = np.linalg.inv(target2cam_R) @ target2base_R
    print(f'cam2base_R:\n{cam2base_R}')

def test6():
    target2cam_xyzrpy = [-0.1628399655335031, 0.10138762431558286, 0.5439752324490061, 0.3631291403893837, -0.008841798828976621, -0.003359646326433152] 
    target2cam_H = xyzrpy_to_H(target2cam_xyzrpy,rad=True)
    
    target2base_xyzrpy = [0.0728359967470169, -0.5675070285797119, 0.4437209963798523, -1.0709999799728394, 0.4970000088214874, -3.072000026702881]
    target2base_H = xyzrpy_to_H(target2base_xyzrpy,rad=True)

    cam2base_H = np.linalg.inv(target2cam_H) @ target2base_H
    
    print(f'cam2base_H:\n{cam2base_H}')

def test7():
    xyzrpy = np.array([0.1,0.2,0.3,170,120,60])
    R = EulerAngle_to_R(xyzrpy[3:6])
    t = xyz_to_t(xyzrpy[0:3])
    H = Rt_to_H(R,t)

    xyzrpy = np.array([0.1, 0.2, 0.3, 0.5, 1.1, 3.0])
    R = Rvec_to_R(xyzrpy[3:6])
    t = xyz_to_t(xyzrpy[0:3])
    H = Rt_to_H(R, t)

    R,t = H_to_Rt(H)
    rpy = R_to_EulerAngle(R)
    rvec = R_to_Rvec(R)
    xyz = t_to_xyz(t)
    xyzrpy = xyz_rpy_to_xyzrpy(xyz,rpy)


if __name__ == '__main__':
    test5()
    test6()
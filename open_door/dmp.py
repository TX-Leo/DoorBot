import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dmp_package.dmp_discrete import dmp_discrete

class DMP():
    def __init__(self,tjt_path):
        self.tjt_dir = os.path.dirname(tjt_path)
        df = pd.read_csv(tjt_path, header=None)
        self.refer_tjt = np.array(df)
        self.data_dim = self.refer_tjt.shape[0] # 6
        self.data_len = self.refer_tjt.shape[1]
        self.dmp = dmp_discrete(n_dmps=self.data_dim, n_bfs=1000, dt=1.0/self.data_len)
        self.dmp.learning(self.refer_tjt)
    
    def gen_new_tjt(self,initial_pos,goal_pos,if_save=True,tjt_save_path=None,img_save_path=None,show=False):
        new_tjt, _, _ = self.dmp.reproduce(initial=initial_pos, goal=goal_pos)
        
        if if_save:
            # save new tjt
            if not tjt_save_path:
                tjt_save_path = f'{self.tjt_dir}/new_tjt.csv'
            df = pd.DataFrame(np.array(new_tjt))
            df.to_csv(tjt_save_path, index=False, header=None)
            
            # save img
            if not img_save_path:
                img_save_path = f'{self.tjt_dir}/dmp.png'
            self.plot_tjt(self.refer_tjt,new_tjt,show=show,save_path=img_save_path)
        
        return new_tjt

    def get_poses(self,tjt,step=50,if_p=False):
        poses = []
        data_dim = tjt.shape[0]
        data_len = tjt.shape[1]
        if data_dim == 6:
            for i in range(0,data_len,step):
                target_pos = tjt[:,i]
                poses.append(target_pos)
            poses.append(tjt[:,-1])
        else:
            for i in range(0,data_dim,step):
                target_pos = tjt[i,:]
                poses.append(target_pos)
            poses.append(tjt[-1,:])
        if if_p:
            print(f'poses:\n{poses}')
        return poses
    
    def get_middle_pose(self,tjt,num=90,if_p=False):
        data_dim = tjt.shape[0]
        data_len = tjt.shape[1]
        if data_dim == 6:
            middle_pos = tjt[:,num]
        else:
            middle_pos = tjt[num,:]
        if if_p:
            print(f'middle_pos:\n{middle_pos}')
        return middle_pos

    def test_random_offset(self,if_initial_offset=True,if_goal_offset=True,show=False):
        x_range = 0.03
        y_range = 0.03
        z_range = 0.03
        rx_range = np.pi/30
        ry_range = np.pi/30
        rz_range = np.pi/30
        if if_initial_offset:
            initial_pos = [self.refer_tjt[0,0] + np.random.uniform(-x_range,x_range), 
                           self.refer_tjt[1,0]  + np.random.uniform(-y_range,y_range), 
                           self.refer_tjt[2,0] + np.random.uniform(-z_range, z_range), 
                           self.refer_tjt[3,0] + np.random.uniform(-rx_range,rx_range), 
                           self.refer_tjt[4,0]  + np.random.uniform(-ry_range,ry_range), 
                           self.refer_tjt[5,0] + np.random.uniform(-rz_range,rz_range)
                           ]
        else:
            initial_pos = [self.refer_tjt[0,0], 
                           self.refer_tjt[1,0], 
                           self.refer_tjt[2,0], 
                           self.refer_tjt[3,0], 
                           self.refer_tjt[4,0], 
                           self.refer_tjt[5,0]
                           ]
        if if_goal_offset:
            goal_pos = [self.refer_tjt[0,-1] + np.random.uniform(-x_range,x_range), 
                        self.refer_tjt[1,-1]  + np.random.uniform(-y_range,y_range), 
                        self.refer_tjt[2,-1] + np.random.uniform(-z_range, z_range), 
                        self.refer_tjt[3,-1] + np.random.uniform(-rx_range,rx_range), 
                        self.refer_tjt[4,-1]  + np.random.uniform(-ry_range,ry_range), 
                        self.refer_tjt[5,-1] + np.random.uniform(-rz_range,rz_range)
                        ]
        else:
            goal_pos = [self.refer_tjt[0,-1], self.refer_tjt[1,-1], self.refer_tjt[2,-1], self.refer_tjt[3,-1], self.refer_tjt[4,-1], self.refer_tjt[5,-1]]
        self.gen_new_tjt(initial_pos,goal_pos,show)

    def plot_tjt(self,refer_tjt,new_tjt,show=False,save_path=None):
        # refer_tjt: 3*n
        # new_tjt: n*3

        fig = plt.figure()

        plt.subplot(2,3,1)
        plt.plot(refer_tjt[0,:], 'g', label='reference')
        plt.plot(new_tjt[:,0], 'r--', label='reproduce')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('points')

        plt.subplot(2,3,2)
        plt.plot(refer_tjt[1,:], 'g', label='reference')
        plt.plot(new_tjt[:,1], 'r--', label='reproduce')
        plt.legend()
        plt.xlabel('y')
        plt.ylabel('points')

        plt.subplot(2,3,3)
        plt.plot(refer_tjt[2,:], 'g', label='reference')
        plt.plot(new_tjt[:,2], 'r--', label='reproduce')
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('points')

        plt.subplot(2,3,4)
        plt.plot(refer_tjt[3,:], 'g', label='reference')
        plt.plot(new_tjt[:,3], 'r--', label='reproduce')
        plt.legend()
        plt.xlabel('rx')
        plt.ylabel('points')

        plt.subplot(2,3,5)
        plt.plot(refer_tjt[4,:], 'g', label='reference')
        plt.plot(new_tjt[:,4], 'r--', label='reproduce')
        plt.legend()
        plt.xlabel('ry')
        plt.ylabel('points')

        plt.subplot(2,3,6)
        plt.plot(refer_tjt[5,:], 'g', label='reference')
        plt.plot(new_tjt[:,5], 'r--', label='reproduce')
        plt.legend()
        plt.xlabel('rz')
        plt.ylabel('points')

        plt.draw()
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
    
def record_tjt(arm,initial_pos,goal_pos,tjt_save_path=None):
    pos_record_x = list()
    pos_record_y = list()
    pos_record_z = list()
    pos_record_rx = list()
    pos_record_ry = list()
    pos_record_rz = list()

    record_enable = False

    while True:
        # get the currten position
        current_pos = arm.get_p()
        print(f'current_pos: {current_pos}')
        
        # if (record_enable == False) and (np.sqrt((current_pos[0] - initial_pos[0])**2 + (current_pos[1] - initial_pos[1])**2 + (current_pos[2] - initial_pos[2])**2) < 0.005):
        if (record_enable == False):
            if (np.sqrt((current_pos[0] - initial_pos[0])**2 + (current_pos[1] - initial_pos[1])**2 + (current_pos[2] - initial_pos[2])**2) > 0.005):
                record_enable = True
                print('find a point')
            else:
                print('wait for moving beyond the initial pos')

        if (np.sqrt((current_pos[0] - goal_pos[0])**2 + (current_pos[1] - goal_pos[1])**2 + (current_pos[2] - goal_pos[2])**2) < 0.005):
            record_enable = False
            print('reach the goal pos')
            break

        if record_enable == True:
            pos_record_x.append(current_pos[0])
            pos_record_y.append(current_pos[1])
            pos_record_z.append(current_pos[2])
            pos_record_rx.append(current_pos[3])
            pos_record_ry.append(current_pos[4])
            pos_record_rz.append(current_pos[5])
            print('record a point')

    print(f'pos number: {len(pos_record_x)}')
    print('Program terminated')

    #save the recorded data to files
    data = np.vstack((pos_record_x, pos_record_y, pos_record_z,pos_record_rx,pos_record_ry,pos_record_rz))
    # print(data)
    df = pd.DataFrame(data)
    df.to_csv(tjt_save_path, index=False, header=None)

if __name__ == '__main__':
    ## record a tjt for dmp
    from arm import Arm
    arm = Arm('192.168.10.19',8080,cam2base_H_path='cfg/cam2base_H.csv',if_gripper=True,if_monitor=False)# 18 for left 19 for right
    
    initial_pos = [0.008727000094950199, -0.1794009953737259, -0.7527850270271301, -3.069000005722046, 0.050999999046325684, -0.5910000205039978]
    goal_pos = [0.41749700903892517, -0.3552910089492798, -0.30075299739837646, -1.3819999694824219, -1.1319999694824219, -1.937999963760376]
    tjt_save_path = 'refer_tjt.csv'
    record_tjt(arm,initial_pos,goal_pos,tjt_save_path)
    
    # ## init dmp
    # dmp = DMP(tjt_save_path)

    # ## test random offset
    # dmp.test_random_offset(if_initial_offset=True,if_goal_offset=True,show=True)

    # ## get new tjt
    # initial_pos = [0.0, -0.12, -0.6, -2.8, 0.04, -0.5]
    # goal_pos = [0.4, -0.37, -0.35, -1.31, -1.19, -1.4]
    # dmp.gen_new_tjt(initial_pos,goal_pos,show=True)


    
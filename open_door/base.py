'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-24 18:04:00
Version: v1
File: 
Brief: 
'''
import socket
import time
import os
import sys
import select
import json
import numpy as np
from utils.lib_io import *

class Base(object):
    def __init__(self,host_ip='192.168.10.10',host_port=31001,linear_velocity=0.2,angular_velocity=0.5):
        self.host_ip = host_ip
        self.host_port = host_port
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

        self.connect()
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_base.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.host_ip,cfg.host_port,cfg.linear_velocity,cfg.angular_velocity)

    def __str__(self):
        print(f'[Base]: host_ip: {self.host_ip}, host_port: {self.host_port}, linear_velocity: {self.linear_velocity}, angular_velocity: {self.angular_velocity}')
        return ''
    
    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('==========\nBase Connecting...')
        self.client_socket.connect((self.host_ip, self.host_port))
        self.start_x,self.start_y,self.start_theta = self.get_location(if_p=False)
        print('Base Connected\n==========')
    
    def move_forward(self,vel=None):
        if not vel:
            vel = self.linear_velocity
        vel = abs(vel)
        command = f"/api/joy_control?angular_velocity=0&linear_velocity={vel}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)

    def move_back(self,vel=None):
        if not vel:
            vel = self.linear_velocity
        vel = -abs(vel)
        command = f"/api/joy_control?angular_velocity=0&linear_velocity={vel}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)

    def move_left(self,vel=None):
        if not vel:
            vel = self.angular_velocity
        vel = abs(vel)
        command = f"/api/joy_control?angular_velocity={vel}&linear_velocity=0"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)

    def move_right(self,vel=None):
        if not vel:
            vel = self.angular_velocity
        vel = -abs(vel)
        command = f"/api/joy_control?angular_velocity={vel}&linear_velocity=0"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)
    
    def move(self,linear_velocity=None,angular_velocity=None):
        if not linear_velocity:
            linear_velocity = self.linear_velocity
        if not angular_velocity:
            angular_velocity = self.angular_velocity
        command = f"/api/joy_control?angular_velocity={angular_velocity}&linear_velocity{linear_velocity}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)

    def move_stop(self,if_p=False):
        command = f"/api/estop"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)
        if if_p:
            print(f'[Base Stop]')

    def move_char(self,char,linear_velocity=None,angular_velocity=None):
        if not linear_velocity:
            linear_velocity = self.linear_velocity
        if not angular_velocity:
            angular_velocity = self.angular_velocity
        if char == "w" or char == "H":
            self.move_forward(vel=linear_velocity)
        elif char == "s" or char == "P":
            self.move_back(vel=linear_velocity)
        elif char == "a" or char == "K":
            self.move_left(vel=angular_velocity)
        elif char == "d" or char == "M":
            self.move_right(vel=angular_velocity)
        elif char == "x":  # Stop
            self.move_forward(0)
        else:
            pass  # Ignore other keys

    def move_T(self,T,linear_velocity=None,if_p=False):
        if not linear_velocity:
            linear_velocity = self.linear_velocity
        start_time = time.time()
        num = 0
        while True:
            if T<=0:
                self.move_char(char='s',linear_velocity=linear_velocity,angular_velocity=0)
            else:
                self.move_char(char='w',linear_velocity=linear_velocity,angular_velocity=0)
            num+=1
            time.sleep(0.01)
            if if_p:
                print(f'[Time]: {time.time() - start_time}')
            if time.time() - start_time > abs(T):
                self.move_stop()
                break
    
    def rotate_T(self,T,angular_velocity=None,if_p=False):
        if not angular_velocity:
            angular_velocity = self.angular_velocity
        start_time = time.time()
        num = 0
        while True:
            if T<=0:
                self.move_char(char='a',linear_velocity=0,angular_velocity=angular_velocity)
            else:
                self.move_char(char='d',linear_velocity=0,angular_velocity=angular_velocity)
            num+=1
            time.sleep(0.01)
            if if_p:
                print(f'[Time]: {time.time() - start_time}')
            if time.time() - start_time > abs(T):
                self.move_stop()
                break
            
    def move_open_door(self,T,linear_velocity=None,angular_velocity=None,if_p=False):
        start_time = time.time()
        num = 0
        if not linear_velocity:
            linear_velocity = self.linear_velocity
        if not angular_velocity:
            angular_velocity = self.angular_velocity
        while True:
            if linear_velocity<0:
                self.move_char(char='s',linear_velocity=linear_velocity,angular_velocity=0)
                time.sleep(0.01)
                self.move_char(char='s',linear_velocity=linear_velocity,angular_velocity=0)
                time.sleep(0.01)
                self.move_char(char='s',linear_velocity=linear_velocity,angular_velocity=0)
                time.sleep(0.01)
            elif linear_velocity>0:
                self.move_char(char='w',linear_velocity=linear_velocity,angular_velocity=0)
                time.sleep(0.01)
                self.move_char(char='w',linear_velocity=linear_velocity,angular_velocity=0)
                time.sleep(0.01)
                self.move_char(char='w',linear_velocity=linear_velocity,angular_velocity=0)
                time.sleep(0.01)
            if angular_velocity<0:
                self.move_char(char='a',linear_velocity=0,angular_velocity=angular_velocity)
            elif angular_velocity>0:
                self.move_char(char='d',linear_velocity=0,angular_velocity=angular_velocity)
            num+=1
            if if_p:
                print(f'[Time]: {time.time() - start_time}')
            if time.time() - start_time > abs(T):
                self.move_stop()
                break
        
    def move_to_door(self,door_plane_weights,offset_in_front=0.6,d2t_coefficient=4.8,linear_velocity=None,if_p=False):
        if not linear_velocity:
            linear_velocity = self.linear_velocity
        D,A,B,C = door_plane_weights
        x,y,z = [0,0,0]
        distance = abs(A * x + B * y + C * z + D) / np.sqrt(A**2 + B**2 + C**2)
        T = (distance-offset_in_front)*d2t_coefficient
        self.move_T(T=T,linear_velocity=linear_velocity)
        if if_p:
            print(f"[distance]: {distance}")
            print(f"[T]: {T}")
        return T,distance

    def move_keyboard_win(self, interval=0.1,linear_velocity=None,angular_velocity=None):
        if not linear_velocity:
            linear_velocity = self.linear_velocity
        if not angular_velocity:
            angular_velocity = self.angular_velocity
        import msvcrt
        def getch():
            char = msvcrt.getch()
            if char == b'\xe0':
                return {
                    b'H': "up",
                    b'P': "down",
                    b'K': "left",
                    b'M': "right",
                }.get(char, None)
            else:
                return char.decode('utf-8') 

        while True:
            try: 
                if msvcrt.kbhit():
                    char = getch()
                    print(f'char: {char}')
                    self.move_char(char,linear_velocity=linear_velocity,angular_velocity=angular_velocity)
                    time.sleep(interval)  # Adjust delay as needed
            except KeyboardInterrupt:  # Allow Ctrl+C to exit
                break

    def move_keyboard_linux(self, interval=0.1,linear_velocity=None,angular_velocity=None):
        if not linear_velocity:
            linear_velocity = self.linear_velocity
        if not angular_velocity:
            angular_velocity = self.angular_velocity
        def getch():
            import sys
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                char = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return char
        while True:
            try: 
                char = getch()
                self.move_char(char,linear_velocity=linear_velocity,angular_velocity=angular_velocity)
                time.sleep(interval)  # Adjust delay as needed
            except KeyboardInterrupt:  # Allow Ctrl+C to exit
                    break
    
    def move_keyboard(self,interval=0.1,linear_velocity=None,angular_velocity=None):
        if not linear_velocity:
            linear_velocity = self.linear_velocity
        if not angular_velocity:
            angular_velocity = self.angular_velocity
        print(f'Start Keyboard Control ...')
        if os.name == 'nt':  # Windows
            self.move_keyboard_win(interval,linear_velocity=linear_velocity,angular_velocity=angular_velocity)
        else:  # Linux
            self.move_keyboard_linux(interval,linear_velocity=linear_velocity,angular_velocity=angular_velocity)

    def move_location(self,location,distance_tolerance=0.5,theta_tolerance=0.05):
        x,y,theta = location
        command = f"/api/move?location={x},{y},{theta}&distance_tolerance={distance_tolerance}&theta_tolerance={theta_tolerance}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)
    
    def move_marker(self, marker_name,distance_tolerance=0.5,theta_tolerance=0.05):
        command = f"/api/move?marker={marker_name}&distance_tolerance={distance_tolerance}&theta_tolerance={theta_tolerance}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)
    
    def get_location(self,if_p=False):
        command = f"/api/robot_status"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(9192).decode()
        # print("response:", response)
        robot_status = json.loads(response)
        robot_status = robot_status['results']
        current_floor = robot_status['current_floor']
        current_pose = robot_status['current_pose']
        x = current_pose['x']
        y = current_pose['y']
        theta = current_pose['theta']
        location = [x,y,theta]
        if if_p:
            print(f'location: {location}')
        return location

    def insert_marker(self,marker_name='1311'):
        command = f"/api/markers/insert?name={marker_name}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)
    
    def check_marker(self,marker_name='1311'):
        command = f"/api/markers/query_list"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(9192).decode()
        # print("response:", response)
        results = json.loads(response)
        results = results['results']
        marker = results[marker_name]
        print(f'{marker_name}: {marker}')

    def disconnect(self):
        self.client_socket.close()

if __name__ == "__main__":
    ## init
    # base = Base.init_from_yaml(cfg_path='cfg/cfg_base.yaml'
    # base = Base(linear_velocity=0.2,angular_velocity=0.2) # slow
    base = Base(linear_velocity=0.8,angular_velocity=0.6) # fast
    print(base)

    ## move keyboard
    base.move_keyboard(interval=0.01)

    # ## move T
    # base.move_T(-0.5,0.2)
    # time.sleep(2)

    ## open door
    # base.move_open_door(T=5,linear_velocity=-0.8,angular_velocity=0,if_p=False) # -0.6

    ## disconnct
    # base.disconnect()
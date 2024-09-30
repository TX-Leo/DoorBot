'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-25 15:34:27
Version: v1
File: 
Brief: 
'''
import paramiko 
import os
import stat
from tqdm import tqdm 

from utils.lib_io import *

class Server():
    def __init__(self,hostname,username,password,if_stfp=True):
        self.hostname = hostname
        self.username = username
        self.password =  password
        self.if_stfp = if_stfp

        self.connect()
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_server.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.hostname,cfg.username,cfg.password,cfg.if_stfp)

    def __str__(self):
        return f'[Server]: hostname: {self.hostname}, username: {self.username}, password: {self.password}, if_stfp: {self.if_stfp}'
        return ''
        
    def connect(self):
        print('==========\nServer Connecting...')
        self.client = paramiko.SSHClient() # Create an SSH client
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # Automatically add the server's host key (this is insecure and should be avoided in production)
        try:
            self.client.connect(hostname=self.hostname, username=self.username, password=self.password)
            if self.if_stfp:
                self.sftp = self.client.open_sftp()
            print('Server Connected\n==========')
        except paramiko.AuthenticationException:
            print("Authentication failed. Please check your credentials.")
        except paramiko.SSHException as ssh_ex:
            print("Error occurred while connecting or establishing an SSH session:", str(ssh_ex))
        except paramiko.ssh_exception.NoValidConnectionsError as conn_ex:
            print("Unable to connect to the server:", str(conn_ex))
        except Exception as ex:
            print("An error occurred:", str(ex))
        
    def disconnect(self):
        self.client.close()
        if self.if_stfp:
            self.sftp.close()
    
    def transfer_file_local2remote(self,local_file_path,remote_file_path,if_p=False,show_process=False):
        self.sftp.put(local_file_path, remote_file_path)

        if show_process:
            file_size = os.path.getsize(local_file_path) 
            with open(local_file_path, 'rb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=f"Upload {os.path.basename(local_file_path)}") as pbar: 
                    def callback(sent, total):
                        pbar.update(sent - pbar.n)
                    sftp.put(local_file_path, remote_file_path, callback=callback) 

        if if_p:
            print(f'{local_file_path} has been transfered to {remote_file_path}')
        
    def transfer_file_remote2local(self,remote_file_path,local_file_path,if_p=False):
        self.sftp.get(remote_file_path, local_file_path)
        if if_p:
            print(f'{remote_file_path} has been transfered to {local_file_path}')
        
    def transfer_folder_remote2local(self,remote_dir, local_dir,if_p=False):
        # Make sure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # List files and directories in the remote path
        files = self.sftp.listdir(remote_dir)

        for file in files:
            remote_file_path = os.path.join(remote_dir, file)
            local_file_path = os.path.join(local_dir, file)
            if stat.S_ISDIR(self.sftp.stat(remote_file_path).st_mode):
                # If it's a directory, recursively call the function
                self.transfer_folder_remote2local(remote_file_path, local_file_path)
            else:
                # If it's a file, download it
                self.sftp.get(remote_file_path, local_file_path)
        if if_p:
            print(f'{remote_dir} has been transfered to {local_dir}')
    
    def create_file_on_remote(self,script:str,remote_script_path,if_p=False):
        with self.sftp.open(remote_script_path, 'w') as remote_file:
            remote_file.write(script)
        if if_p:
            print(f'{remote_script_path} has been created')

    def exec_cmd(self,cmd,if_p=False):
        stdin, stdout, stderr = self.client.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status() # wait for execution to finish
        if exit_status == 0:
            if if_p:
                print("Successfully executed command on server")
        else:
            print(f"Failed to execute command. Error code: {exit_status}")
            error_output = stderr.read().decode().strip()
            print(f"Error output: {error_output}")
        if if_p:
            output = stdout.read().decode('utf-8')
            print(f'the output of server:\n {output}')


if __name__ == "__main__":
    ## init server
    server = Server.init_from_yaml(cfg_path='cfg/cfg_server.yaml')
    print(server)

    ## transfer file from local to remote
    # local_file_path = ''
    # remote_file_path = ''
    # server.transfer_file_local2remote(local_file_path,remote_file_path,if_p=True,show_process=True)

    ## disconnect
    server.disconnect()
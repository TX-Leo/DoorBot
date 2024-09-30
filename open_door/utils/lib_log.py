'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-27 15:40:25
Version: v1
File: 
Brief: 
'''
import os
import logging
from logging.handlers import TimedRotatingFileHandler

import sys
root_dir = '../'
sys.path.append(root_dir)

from utils.lib_io import *

class Logger(object):
    def __init__(self,log_path='./log'):
        mkfile(log_path)
        # self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '%s' % log_path)
        self.log_path = log_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_logger.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.log_path)
    
    def console(self, level, message):
        # Create a TimedRotatingFileHandler for writing to local files
        fh = TimedRotatingFileHandler(self.log_path, when='MIDNIGHT', interval=1, encoding='utf-8')
        fh.suffix = '%Y-%m-%d.log'
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

        # Create a StreamHandler for output to the console
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

        if level == 'info':
            self.logger.info(f"INFO - {message}") 
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)  # Do not display error stack
        elif level == 'error_':
            self.logger.error(message, exc_info=1) # Display error stack
        elif level == 'time':
            self.logger.info(f"TIME - {message}") 
        elif level == 'flag':
            self.logger.info(f"FLAG - {message}") 

        # To avoid duplicate log output
        self.logger.removeHandler(ch)
        self.logger.removeHandler(fh)

        fh.close()

    def debug(self, message):
        self.console('debug', message)

    def info(self, message):
        self.console('info', message)

    def warning(self, message):
        self.console('warning', message)

    def error(self, message):
        self.console('error', message)

    def error_(self, message):
        self.console('error_', message)

    def time(self, message):
        self.console('time', message)
    
    def flag(self, message):
        self.console('flag', message)

if __name__ == '__main__':
    logger = Logger.init_from_yaml(cfg_path='cfg/cfg_logger.yaml')
    logger.info(f'Set_Joint_Speed:{1+1}')
    logger.warning(f'Set_Joint_Speed:{1+1}')
    logger.error(f'Set_Joint_Speed:{1+1}')
    logger.error_(f'Set_Joint_Speed:{1+1}')

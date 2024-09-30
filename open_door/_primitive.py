'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-27 15:55:31
Version: v1
File: 
Brief: 
'''
class _Primitive:
    def __init__(self, action, id, ret, param, error):
        self.action = action
        self.id = id
        self.ret = ret
        self.param = param
        self.error = error
    
    def to_list(self):
        return [self.action, self.id, self.ret, self.param, self.error]
    
    def __str__(self):
        print(f'action: {self.action}, id: {self.id}, ret: {self.ret}, param: {self.param}, error: {self.error}')
        return ''
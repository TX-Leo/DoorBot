'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-24 18:14:07
Version: v1
File: 
Brief: Execute primitives step by step. 
'''
import argparse
import sys
root_dir = './'
sys.path.append(root_dir)
from primitive import Primitive

def run(tjt_num,type):
    primitive = Primitive(root_dir=root_dir,tjt_num=tjt_num,type=type)
    primitive.run()

def main(args):
    run(tjt_num=args.tjt_num,type=args.type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--tjt_num",type=int,default=0,help="trajectory number.")
    parser.add_argument("-t","--type",type=str,default="lever",help="type of handle (lever / knob / crossbar / drawer).")
    main(parser.parse_args())
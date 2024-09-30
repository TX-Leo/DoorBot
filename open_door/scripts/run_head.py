import sys
root_dir = "../"
sys.path.append(root_dir)

from head import Head

head = Head.init_from_yaml(cfg_path=f'{root_dir}/cfg_head.yaml')

print(head)

head.servo_move(1000,1,servo_1_position=400)
head.servo_move(1000,2,servo_2_position=500)

head.disconnect()
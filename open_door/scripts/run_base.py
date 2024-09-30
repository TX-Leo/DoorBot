import sys
root_dir = "../"
sys.path.append(root_dir)

from base import Base

base = Base.init_from_yaml(cfg_path=f'{root_dir}/cfg_base.yaml')

print(base)

base.move_keyboard(interval=0.01)

base.disconnect()
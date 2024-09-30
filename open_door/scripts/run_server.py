import sys
root_dir = "../"
sys.path.append(root_dir)

from server import Server

server = Server.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_server.yaml')

print(server)

server.disconnect()
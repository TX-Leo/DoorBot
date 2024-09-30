import sys
root_dir = "../"
sys.path.append(root_dir)

from camera import Camera

camera = Camera.init_from_yaml(cfg_path=f'{root_dir}/cfg_cam.yaml')

print(camera)

camera.rgbd_viewer()

camera.disconnect()
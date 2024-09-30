RANSAC for detecing planes from depth image
=======================================


## Description
It's the RANSAC python package for detect planes from depth image.

The main scripts are: [plane_detector.py](plane_detector.py) and [config/plane_detector_config.yaml](config/plane_detector_config.yaml).

Specifically, it contains the following steps:

```
(1) Create point cloud from depth image
(2) while RANSAC hasn't failed:
(3)    Use RANSAC to detect a plane from point cloud.
(4)    Remove the plane's points from point cloud.
```

## Installation
```
pip install -r requirements.txt
```

## Usage
Run `run_plane_detector.sh`
There are four parameters:
- `CF`: The RANSAC configuration file path.
- `CAMERA`: The camera configuration file path.
- `V`: If visualization is enabled.
- `O`: The orientation of the handle (vertical or horizontal)

You need to change the parameters in the configuration file carefully.

More details can be checked in [ransac.py](../ransac.py)

## Acknowledgements
Thank [Feiyu Chen](https://github.com/felixchenfy) for providing the initial [code](https://github.com/felixchenfy/ros_detect_planes_from_depth_img).
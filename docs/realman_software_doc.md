<!--
 * @Author: TX-Leo
 * @Mail: tx.leo.wz@gmail.com
 * @Date: 2024-09-18 23:24:17
 * @Version: v1
 * @File: 
 * @Brief: 
-->
## RealMan Robot Usage
You can run codes in the [scripts](../open_door/scripts/) to control different parts of the robot.

Please check out [cfg](../open_door/cfg) which contains all configuration files before running the scripts.

- [run_arm_left](../open_door/scripts/run_arm_left.py)
- [run_arm_right](../open_door/scripts/run_arm_right.py)
- [run_base](../open_door/scripts/run_base.py)
- [run_camera](../open_door/scripts/run_camera.py)
- [run_head](../open_door/scripts/run_head.py)

## Modules
- [run_clip](../open_door/scripts/run_clip.py)
- [run_dmp](../open_door/scripts/run_dmp.py)
- [run_dtsam](../open_door/scripts/run_dtsam.py)
- [run_gemini](../open_door/scripts/run_gemini.py)
- [run_gum](../open_door/scripts/run_gum.py)
- [run_server](../open_door/scripts/run_server.py)

## Code Base
- arm
  - [arm_package](../open_door/arm_package)
  - [arm.py](../open_door/arm.py)
- dmp (Dynamic Movement Primitives (DMPs))
  - [dmp_package](../open_door/dmp_package)
  - [dmp.py](../open_door/dmp.py)
- dtsam (Detic + SAM)
  - [dtsam_package](../open_door/dtsam_package)
  - [dtsam.py](../open_door/dtsam.py)
- ransac (Random Sample Consensus (RANSAC))
  - [ransac_package](../open_door/ransac_package)
  - [ransac.py](../open_door/ransac.py)
- gum (Grasping-and-Unlocking Model (GUM))
  - [gum_package](../open_door/gum_package)
  - [gum.py](../open_door/gum.py)


## DEBUG
### Problem: AttributeError: module 'clip' has no attribute 'load'
- pip install git+https://github.com/openai/CLIP.git
- reference: https://github.com/openai/CLIP/issues/180

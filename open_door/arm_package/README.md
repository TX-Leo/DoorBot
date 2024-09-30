Realman Bimanual Humanoid Robot Python3 API
=======================================

## Description
This is the python API code for controlling two arms of the realman robot.

The main script is: [robotic_arm.py](robotic_arm.py).


## Installation
Before controlling the robot, you must conduct camera calibration and handeye calibration.
- Camera calibration can be performed by MatLab Camera Calibration Toolbox.
- Details about handeye calibration can be checked in [handeye_calibration](handeye_calibration) folder.

Also, to use this API, you need to install the following dependencies:
- In Windows system, you need `RM_Base.dll`
- In Linux system, you need `libRM_Base.so`

## Usage
To use the API, you need to import the `robotic_arm` module and create an instance of the `Arm` class.

Details can be check in [arm.py](../arm.py)

## Acknowledgements
Thank RealMan Service People for providing support and the initial API code.
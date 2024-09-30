DMPs Python Package
=======================================

## Description
It is the DMPs (Dynamic Motion Primitives) python package for control the robotic arm moving following the reference manner.

The main scripts are: [dmp_discrete.py](dmp_discrete.py) and [record_trajectory_xyzrxryrz.py](record_trajectory_xyzrxryrz.py]).

## Installation
```
pip install -r requirements.txt
```

## Usage
- Specifically, you should run `record_trajectory_xyzrxryrz.py` to generate a reference trajectory (record position(xyz) and orientation(rpy) of the end-effector for each time step). 
It can be the generated trajectory by human interaction or pre-defined trajectory.

- Then, you can reproduce a new trajectory by providing the starting point and ending point. The starting point can be the current position of the end-effector, and the ending point can be the desired position of the end-effector. The new trajectory is trying to learn the pattern of the reference trajectory like some ups and downs.

- And then you can let the robot move along the new trajectory by position-control.

You can check the details in `open_door/dmp.py`

## Acknowledgements
Thank [Chaobin Zou](https://github.com/chauby) for providing the initial [code](https://github.com/chauby/PyDMPs_Chauby).
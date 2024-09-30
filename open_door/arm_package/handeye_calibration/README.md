<!--
 * @Author: TX-Leo
 * @Mail: tx.leo.wz@gmail.com
 * @Date: 2024-07-19 02:02:18
 * @Version: v1
 * @File: 
 * @Brief: 
-->

## Step1: Data Collection

1. **Prepare the Setup:**
    - Securely attach the grid board to the robot gripper. 
    - Ensure you know the physical dimensions and grid spacing of the board.
    - Maintain a fixed relative position between the gripper and the grid board throughout the data collection process.

2. **Capture Images:**
    - Run `data_collection.py`.
    - Control the robot arm to move to various positions and orientations within the camera's field of view.
    - At each stable position, trigger the RealSense camera to capture an image of the grid board.
    - Aim to collect 40-60 images, strategically covering a wide range of 3D positions for improved calibration accuracy.

## Step2: Hand-Eye Calibration

1. **Run Calibration Script:**
    - Execute `handeye_calibration.py`.

2. **Automatic Processing:**
    - The script will automatically detect the grid pattern within each captured image.
    - Using the detected patterns and corresponding robot arm poses, the script will calculate the transformation matrix between the camera and the robot arm.

3. **Output:**
    - The script will output a 4x4 hand-eye calibration matrix (H). This matrix encapsulates the translation and rotation information required to transform coordinates between the camera frame and the robot arm base frame. 
def gen_hl_prompt(last_pmt_action, last_pmt_error):
    hl_prompt = f"""
        ### context:
        ===
        I want to enable my bimanual humanoid robot to learn how to manipulate articulated objects, specifically focusing on four types: doors with lever handle, doors with doorknob, doors with crossbar, and drawers.

        ===
        To achieve this, I've designed a set of primitives that constitute the robot's action space: premove, grasp, unlock, rotate, open, swing, home, and back. Each primitive represents a specific action the robot can perform. 
        The input to a primitive consists of parameters, and the output is an error type, indicating the outcome of the action.

        ===
        Here are the definitions of each primitive:

        1. **Premove**: The robot's base positions itself at a predefined distance in front of the target object (e.g., door). You should call this when the robot is not close enought to the door accroding to the given image.
        2. **Grasp**: The robot attempts to grasp the handle of the target object using its one arm and one gripper.
        3. **Unlock**:  (Applicable to lever handles only) The robot rotates the lever handle to an unlocked position using its gripper.
        4. **Rotate**: (Applicable to doorknobs only) The robot rotates the doorknob to an unlocked position using its gripper.
        5. **Open**: The robot's base moves forward or back to open the target object (e.g., pushes or pulls a door).
        6. **Swing**: The robot uses its other arm to assist in fully opening the target object.
        7. **Home**: The robot moves the manipulating arm back to its default home position (A naturally drooping state).
        8. **Back**: The robot moves the manipulating arm back to the grasping position for **Grasp** primitive. 

        ===
        Here are the parameter descriptions for each primitive:

        1. **Premove**: No parameters.
        2. **Grasp**: [dx, dy, R]
            **dx,dy**: Grasping point offset. The initial grasping point (Cx, Cy) is determined by the center of the handle's mask in the 2D image. The final grasping point, P1, is calculated as (Cx + dx, Cy + dy). The offset (dx, dy) allows for fine-tuning the grasp position to ensure a secure grip on the handle.
            **R**: Unlocking rotation radius (applicable to lever handles only, set to zero for other handle types). This parameter defines the radius of the circular arc used for unlocking the lever handle. The point P1 is rotated by 90 degrees around the handle's axis, with a radius of R, resulting in point P2. The robot arm then moves to P2 to execute the unlocking motion. The R value should be chosen to ensure a smooth and effective unlocking trajectory based on the handle's geometry.
        3. **Unlock**: No parameters.
        4. **Rotate**: No parameters.
        5. **Open**: No parameters.
        6. **Swing**: No parameters.
        7. **Home**: No parameters.
        8. **Back**: No parameters.

        ===
        Here are the output (error type) descriptions for each primitive:

        1. **SUCCESS**: The primitive execution was successful.
        2. **GRASP_SAFETY**: A safety constraint was violated during the grasp attempt.
        3. **GRASP_IK_FAIL**: The robot's inverse kinematics solver failed to find a valid solution to reach the grasp target.
        4. **GRASP_MISS**: The robot failed to grasp the handle.
        5. **UNLOCK_SAFETY**: A safety constraint was violated during the unlock attempt.
        6. **UNLOCK_MISS**: The robot failed to maintain its grasp on the handle while unlocking.
        7. **ROTATE_SAFETY**: A safety constraint was violated during the rotation attempt.
        8. **ROTATE_MISS**: The robot failed to maintain its grasp on the handle while rotating.
        9. **OPEN_SAFETY**: A safety constraint was violated during the opening attempt.
        10. **OPEN_MISS**: The robot failed to maintain its grasp on the handle while opening. 

        ===
        The overall manipulation process is divided into two levels of policy: high-level and low-level. 
        The high-level policy is responsible for selecting the appropriate primitive to execute at each step, while the low-level policy generates the specific parameters required for the chosen primitive.

        ### objective:
        I'm seeking your assistance in defining the high-level policy. 
        Given an image of the robot current state which is captured by robot's camera (if any), and the information about the last executed primitive, including its type and the return error (if any), I need you to determine the next primitive to execute.
        The last primitve info:
        1. **type**: {last_pmt_action}
        2. **return**: {last_pmt_error}

        ### style:
        N/A

        ### tone:
        Precise and concise.

        ### audience:
        This is intended for programming a robot to successfully open various types of doors and drawers.

        ### response:
        Based on the information provided, please respond with the name of the next primitive to execute: premove, grasp, unlock, rotate, open, swing, home, or back. Do not output any "**". 

    """
    return hl_prompt


def gen_ll_prompt(example_param1, example_param2):
    ll_prompt = f"""
        ### context:
        ===
        I want to enable my bimanual humanoid robot to learn how to manipulate articulated objects, specifically focusing on four types: doors with lever handle, doors with doorknob, doors with crossbar, and drawers.

        ===
        To achieve this, I've designed a set of primitives that constitute the robot's action space: premove, grasp, unlock, rotate, open, swing, home, and back. Each primitive represents a specific action the robot can perform. 
        The input to a primitive consists of parameters, and the output is an error type, indicating the outcome of the action.

        ===
        Here are the definitions of each primitive:

        1. **Premove**: The robot's base positions itself at a predefined distance in front of the target object (e.g., door). You should call this when the robot is not close enought to the door accroding to the given image.
        2. **Grasp**: The robot attempts to grasp the handle of the target object using its one arm and one gripper.
        3. **Unlock**:  (Applicable to lever handles only) The robot rotates the lever handle to an unlocked position using its gripper.
        4. **Rotate**: (Applicable to doorknobs only) The robot rotates the doorknob to an unlocked position using its gripper.
        5. **Open**: The robot's base moves forward or back to open the target object (e.g., pushes or pulls a door).
        6. **Swing**: The robot uses its other arm to assist in fully opening the target object.
        7. **Home**: The robot moves the manipulating arm back to its default home position (A naturally drooping state).
        8. **Back**: The robot moves the manipulating arm back to the grasping position for **Grasp** primitive. 

        ===
        Here are the parameter descriptions for each primitive:

        1. **Premove**: No parameters.
        2. **Grasp**: [dx, dy, R]
            **dx,dy**: Grasping point offset. The initial grasping point (Cx, Cy) is determined by the center of the handle's mask in the 2D image. The final grasping point, P1, is calculated as (Cx + dx, Cy + dy). The offset (dx, dy) allows for fine-tuning the grasp position to ensure a secure grip on the handle.
            **R**: Unlocking rotation radius (applicable to lever handles only, set to zero for other handle types). This parameter defines the radius of the circular arc used for unlocking the lever handle. The point P1 is rotated by 90 degrees around the handle's axis, with a radius of R, resulting in point P2. The robot arm then moves to P2 to execute the unlocking motion. The R value should be chosen to ensure a smooth and effective unlocking trajectory based on the handle's geometry.
        3. **Unlock**: No parameters.
        4. **Rotate**: No parameters.
        5. **Open**: No parameters.
        6. **Swing**: No parameters.
        7. **Home**: No parameters.
        8. **Back**: No parameters.

        ===
        Here are the output (error type) descriptions for each primitive:

        1. **SUCCESS**: The primitive execution was successful.
        2. **GRASP_SAFETY**: A safety constraint was violated during the grasp attempt.
        3. **GRASP_IK_FAIL**: The robot's inverse kinematics solver failed to find a valid solution to reach the grasp target.
        4. **GRASP_MISS**: The robot failed to grasp the handle.
        5. **UNLOCK_SAFETY**: A safety constraint was violated during the unlock attempt.
        6. **UNLOCK_MISS**: The robot failed to maintain its grasp on the handle while unlocking.
        7. **ROTATE_SAFETY**: A safety constraint was violated during the rotation attempt.
        8. **ROTATE_MISS**: The robot failed to maintain its grasp on the handle while rotating.
        9. **OPEN_SAFETY**: A safety constraint was violated during the opening attempt.
        10. **OPEN_MISS**: The robot failed to maintain its grasp on the handle while opening. 

        ===
        The overall manipulation process is divided into two levels of policy: high-level and low-level. 
        The high-level policy is responsible for selecting the appropriate primitive to execute at each step, while the low-level policy generates the specific parameters required for the chosen primitive.

        ### objective:
        I'm seeking your assistance in defining the level-level policy. Given an image of a door handle, your task is to:
            1. Identify the handle type: Determine whether the handle is a lever, doorknob, or crossbar.
            2. Generate grasp parameters: Calculate the following parameters for the "GRASP" action:
                dx, dy (pixels): Offsets from the center of the handle in the image. These values can be positive or negative, indicating shifts in the x (horizontal) and y (vertical) directions, respectively.
                R (pixels): Unlocking rotation radius. This parameter is only relevant for lever handles. It represents the radius of the circular arc used for unlocking the lever and can be positive or negative. For other handle types, set R to 0.
            Your parameter selection should prioritize a stable grasp, taking into account the handle's geometry and ensuring a secure grip for manipulation.
        Example Demonstrations: To illustrate the expected output, you are provided with two example images and their corresponding grasp parameters:
            Example Image 1: dx = {{{example_param1[0]}}}, dy = {{{example_param1[1]}}}, R = {{{example_param1[2]}}}
            Example Image 2: dx = {{{example_param2[0]}}}, dy = {{{example_param2[1]}}}, R = {{{example_param2[2]}}}
        Your Task:
            Analyze the provided third image and generate the optimal grasp parameters (dx, dy, R) following the guidelines above.

        ### style:
        N/A

        ### tone:
        Precise and concise.

        ### audience:
        This is intended for programming a robot to successfully open various types of doors and drawers.

        ### response:
        Based on the information provided, please provide your response in the following JSON format:
            type: handle_type, 
            dx: dx_value,
            dy: dy_value,
            R:  r_value
        Where:
        handle_type: The identified handle type (e.g., lever, doorknob, crossbar).
        dx_value: The calculated dx value (as a float number).
        dy_value: The calculated dy value (as a float number).
        r_value: The calculated R value (as a float number).

    """
    return ll_prompt


def gen_detection_prompt(text_prompt):
    detection_prompt = f"""

    ### context:
    You will be provided with an image and two text prompts.

    ### objective:
    Determine the probability of the image being more similar in content to each of the provided text prompts: {text_prompt}.

    ### style:
    N/A

    ### tone:
    Precise and concise.

    ### audience:
    This is intended for programming a robot to successfully open various types of doors and drawers.

    ### response:
    Return your analysis as a list containing two decimal numbers, [prob1, prob2].
    prob1: Represents the probability (between 0.0 and 1.0) of the image being closer to the first text prompt.
    prob2: Represents the probability (between 0.0 and 1.0) of the image being closer to the second text prompt.
    The sum of prob1 and prob2 must always equal 1.0.
    Please directly give me the list, no explanation is needed.

    """
    return detection_prompt
import pybullet as p
import os
import numpy as np
import time
from time import sleep
import matplotlib.pyplot as plt

def calculate_trajectory_speed(points, max_speed, robotID, end_effector_link_index):
    joint_configs = []
    desired_orientation = p.getQuaternionFromEuler([0, np.pi, 0])  

    for i in range(len(points) - 1):
        start_point = np.array(points[i])
        end_point = np.array(points[i + 1])
        distance = np.linalg.norm(end_point - start_point)
        time_required = distance / max_speed
        num_steps = int(time_required / 0.0041)  # Simulation timestep of 0.01s

        # Linear interpolation between points
        for step in range(num_steps):
            t = step / num_steps
            interp_point = (1 - t) * start_point + t * end_point
            joint_angles = p.calculateInverseKinematics(robotID, end_effector_link_index, interp_point.tolist(), desired_orientation)
            joint_configs.append((joint_angles, interp_point))
    return joint_configs

def get_end_effector_position(robot_id, link_index):
    # Get the position part of the link state
    return np.array(p.getLinkState(robot_id, link_index)[4])

def get_end_effector_position2(robot_id, link_index):
    # Get the end effector position in the world frame
    end_effector_state = p.getLinkState(robot_id, end_effector_link_index)
    end_effector_position_world = end_effector_state[0]

    return np.array(end_effector_position_world)

# Connect to PyBullets physics server
physicsClient = p.connect(p.GUI)

prev_position = None  # To store the previous position for speed calculation
speeds = []  # To store the speeds at each step
times = []

robot_urdf_path = os.path.join(os.path.expanduser("~"), "Documents", "UoS", "Year 4", "Individual Project", "PandaRobot.jl", "deps", "Panda", "panda.urdf")
table_urdf_path = os.path.join(os.path.expanduser("~"), "Documents", "UoS", "Year 4", "Individual Project", "bullet3", "data", "table", "table.urdf")

# Load URDF file
tableID = p.loadURDF(table_urdf_path, [0, 0, 0], useFixedBase=True)
robotID = p.loadURDF(robot_urdf_path, [0, 0.2, 0.6], p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True)

end_effector_link_index = 8
num_joints = p.getNumJoints(robotID)
ik_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Adjust this if not all joints are involved

collision_indices = [num_joints - 7,  num_joints - 3]  # Last 4 links
collision_dimensions = [(0.15,0.25), (0.15, 0.1)]

visual_shapes = []
for index, (radius, height) in zip(collision_indices, collision_dimensions):
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CAPSULE, radius=radius, length=height, rgbaColor=[1, 0, 0, 0.2])
    # Get initial link state to position the visual correctly
    link_state = p.getLinkState(robotID, index)
    visual_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=link_state[4], baseOrientation=link_state[5])
    visual_shapes.append((visual_id, index))

# Define target positions and max speed
target_positions = [
    [7.15041687e-13, 2.87999988e-01, 1.52600002e+00], [-2, 1.6, 2.5], [2, 1.6, 2.5], [2, 1.6, 1],
    [0, 1.6, 2.5], [0, 1.6, 1], [2, 1.6, 1], [-2, 1.6, 1], [2, 1.6, 1.3], 
    [-2, 1.6, 1.3], [2, 1.6, 1.6], [-2, 1.6, 1.6], [2, 1.6, 1], [-2, 1.6, 2.5], 
    [2, 1.6, 2.5], [2, 1.6, 1], [0, 1.6, 2.5], [0, 1.6, 1], [2, 1.6, 1], 
    [-2, 1.6, 1], [2, 1.6, 1.3], [-2, 1.6, 1.3], [2, 1.6, 1.6], [-2, 1.6, 1.6] 
]

target_positions2 = [
    [7.15041687e-13, 2.87999988e-01, 1.52600002e+00], [-0.55, 0.6, 1.3], [0.55, 0.6, 1.3], [0.55, 0.6, 1],
    [0, 0.6, 1.2], [-0.55, 0.6, 1], [0.55, 0.6, 1], [-0.55, 0.6, 1], [0.55, 0.6, 1.1], 
    [-0.55, 0.6, 1.1], [0.55, 0.6, 1.2], [-0.55, 0.6, 1.2], [0.55, 0.6, 1], [-0.55, 0.6, 1.3], 
    [0.55, 0.6, 1.3], [0.55, 0.6, 1], [0, 0.6, 1.3], [0, 0.6, 1], [0.55, 0.6, 1], 
    [-0.55, 0.6, 1], [0.55, 0.6, 1.1], [-0.55, 0.6, 1.1], [0.55, 0.6, 1.2], [-0.55, 0.6, 1.2] 
]

max_speed = 1  # m/s
step_counter = 0

joint_trajectory = calculate_trajectory_speed(target_positions, max_speed, robotID, end_effector_link_index)
start_check = False

# Run the trajectory
try:
    for joint_config, target_point in joint_trajectory:
        p.setJointMotorControlArray(robotID, ik_joint_indices, p.POSITION_CONTROL, targetPositions=joint_config)
        p.stepSimulation()
        sleep(1./240.)
        position = get_end_effector_position(robotID, end_effector_link_index)
        print(position)
        for visual_id, link_index in visual_shapes:
            link_state = p.getLinkState(robotID, link_index)
            p.resetBasePositionAndOrientation(visual_id, link_state[4], link_state[5])

        step_counter += 1

        if step_counter >= 45:
            # Get the current position of the end effector
            current_position = get_end_effector_position(robotID, end_effector_link_index)
            print(current_position)

            # Calculate speed if it's not the first step
            if prev_position is not None:
                time_elapsed = time.time() - start_time
                times.append(time_elapsed)

                speed = np.linalg.norm(current_position - prev_position) / (1./120.)  # Assuming time step of 1/240
                speeds.append(speed)
        
            # Update the previous position
            prev_position = current_position

            if start_check == False:
                start_time = time.time()  # Start time of the task
                start_check = True

        sleep(1./240.)  # Slow down the loop to real-time if desired
except KeyboardInterrupt:
    print("Simulation stopped manually")
finally:
    task_time = time.time() - start_time
    print(f"Total task completion time: {task_time:.2f} seconds")

    plt.figure()
    plt.plot(times, speeds, label='End-effector speed')
    plt.xlabel('Simulation time elapsed (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('End-effector speed over Simulation time')
    plt.legend()
    plt.show()

    p.disconnect()

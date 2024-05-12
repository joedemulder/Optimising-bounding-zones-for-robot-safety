import pybullet as p
import os

import numpy as np
import time
from time import sleep

import matplotlib.pyplot as plt
import csv

# Functions

# Function to print joint information
def print_joint_info(robot_id):
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print("Joint Index:", joint_info[0])
        print("Joint Name:", joint_info[1].decode("utf-8"))
        print("Joint Type:", joint_info[2])
        print("First position index:", joint_info[3])
        print("First velocity index:", joint_info[4])
        print("Link Name:", joint_info[12].decode("utf-8"))
        print("----------")

def calculate_trajectory_speed(start_point, end_point, max_speed, robotID, end_effector_link_index):
    joint_configs = []
    if max_speed == 0:
        # Fetch current joint angles and hold these positions for a number of steps
        current_joint_angles = [p.getJointState(robotID, i)[0] for i in range(p.getNumJoints(robotID)-2)]
        num_stop_steps = 10000  # Define the number of simulation steps to maintain the current position
        joint_configs = [current_joint_angles] * num_stop_steps
    else:
        desired_orientation = p.getQuaternionFromEuler([0, np.pi, 0]) 
        distance = np.linalg.norm(np.array(end_point) - np.array(start_point)) 
        time_required = distance / max_speed
        num_steps = int(time_required / 0.0041)  

        # Linear interpolation between points
        for step in range(num_steps):
            t = step / num_steps
            interp_point = (1 - t) * np.array(start_point) + t * np.array(end_point)
            joint_angles = p.calculateInverseKinematics(robotID, end_effector_link_index, interp_point.tolist(), desired_orientation)
            joint_configs.append(joint_angles)
    return joint_configs

def calculate_repulsive_force(current_position, obstacle_position):
    k_rep=0.005
    print("Current position")
    print(current_position)
    print("Obstacle position")
    print(obstacle_position)

    # Calculate the distance vector between the end effector and the obstacle
    distance_vector = np.array(current_position) - np.array(obstacle_position)
    print(distance_vector)
    distance = np.linalg.norm(distance_vector)
    
    # Calculate the magnitude of the repulsive force
    force_magnitude = k_rep * (1/distance) / (distance**2)
    # Normalize the distance vector and scale by the force magnitude
    repulsive_force = force_magnitude * (distance_vector / distance)
    print(repulsive_force)
    return repulsive_force
    

def get_end_effector_position(robot_id, link_index):
    # Get the end effector position in the world frame
    end_effector_state = p.getLinkState(robot_id, link_index)
    end_effector_position_world = end_effector_state[0]

    return np.array(end_effector_position_world)

def calculate_human_trajectory(target_joint_positions, max_speed):
    num_positions = len(target_joint_positions)
    trajectories = [[] for _ in range(len(target_joint_positions[0]))]  # Initialize list of trajectories for each joint

    for i in range(num_positions - 1):
        start_angles = target_joint_positions[i]
        end_angles = target_joint_positions[i + 1]
        # Calculate distance as a simple sum of angle differences for simplicity
        distance = sum(abs(end - start) for start, end in zip(start_angles, end_angles))
        time_required = distance / max_speed
        num_steps = max(int(time_required / 0.01), 1)  # Ensure at least one step

        # Linear interpolation for each joint
        for j, (start, end) in enumerate(zip(start_angles, end_angles)):
            interpolated = np.linspace(start, end, num_steps)
            trajectories[j].extend(interpolated)

    return np.array(trajectories)  # Each row corresponds to one joint's full trajectory

def adjust_box_dynamically(robot_id, link_id, box_id, prev_half_extents):
    # Get the current state of the link
    link_state = p.getLinkState(robot_id, link_id, computeLinkVelocity=1)
    link_pos = link_state[4]  # Base position of the link
    link_orn = link_state[5]  # Orientation of the link
    link_vel = np.array(link_state[6])  # Linear velocity of the link

    # Calculate speed and normalize the velocity vector
    speed = np.linalg.norm(link_vel)
    direction = link_vel / speed 

    # Calculate new half extents based on movement factors
    base_half_extents = np.array([0.1, 0.1, 0.1])  # Base dimensions of the box
    new_half_extents = base_half_extents * (1 + 0.5 * np.abs(direction))

    # Calculate the new position offsets
    offsets = base_half_extents * 1.2 * direction
    new_position = np.array(link_pos) + offsets

    change_threshold = 0.01  # Threshold to decide when to recreate the shape
    if np.any(np.abs(new_half_extents - prev_half_extents) > change_threshold):
        # Update the box dimensions and position
        p.removeBody(box_id)
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=new_half_extents)
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=new_half_extents, rgbaColor=[1, 0, 0, 0.4])
        box_id = p.createMultiBody(baseMass=0.000001,
                                    baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id,
                                    basePosition=new_position,
                                    baseOrientation=link_orn
        )

        enableCollision = 0
        p.setCollisionFilterPair(box_id, humanID, -1, -1, enableCollision)

        p.resetBasePositionAndOrientation(box_id, new_position, link_state[5])
        
    else:
        # Only update position and orientation
        p.resetBasePositionAndOrientation(box_id, new_position, link_state[5])

    return box_id, new_half_extents

# - - - - - - - - - - - - - - - - - -

# Simulation initialisation

physicsClient = p.connect(p.GUI) # Connect to PyBullets physics server
p.setTimeStep(1./240.) # Set timestep to 1/240 (approx 0.0041s)

# - - - - - - - - - - - - - - - - - -

# Simulation objects initialisation

# Load models + set positions
robot_urdf_path = os.path.join(os.path.expanduser("~"), "Documents", "UoS", "Year 4", "Individual Project", "PandaRobot.jl", "deps", "Panda", "panda.urdf")
table_urdf_path = os.path.join(os.path.expanduser("~"), "Documents", "UoS", "Year 4", "Individual Project", "bullet3", "data", "table", "table.urdf")
human_urdf_path = os.path.join(os.path.expanduser("~"), "Documents", "UoS", "Year 4", "Individual Project", "human-gazebo-master", "humanSubject01", "humanSubject01_48dof.urdf")
# Load URDF files
tableID = p.loadURDF(table_urdf_path, [0, 0, 0], useFixedBase=True)
robotID = p.loadURDF(robot_urdf_path, [0, 0.2, 0.6], p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True)
# Load human model facing the robot, adjust the orientation as needed
humanID = p.loadURDF(human_urdf_path, [0, 1.5, 0.9], p.getQuaternionFromEuler([0, 0, -np.pi/2]), useFixedBase=True)

# - - - - - - - - - - - - - - - - - -

# Arm position initialisation

right_arm_joints = [14, 16, 17, 19, 24]  # Indices to single out right arm movement
arm_speed = 0.8 # Speed for arm movement
hand_index = 21

# Arm joint angles for three arm movement patterns
target_arm_angles_static = [[0.2, 0.4, 1.5, 0.2, -1.2], [0.2, 0.4, 1.5, 0.2, -1.2]]*30000
target_arm_angles_basic = [[0.2, 0.4, 1.5, 0.2, -1.2], [0.2, 1.2, 1.5, 0.2, -1.2]]*1000
target_arm_angles_complex = [[0.2, 0.4, 1.5, 0.2, -1.2],[0.2, -0.5, 1.2, 0.2, -1.2],[0.2, 1, 1.7, 0.2, -1.2],
                             [0.2, 0.8, 1.7, 1, -1.2], [0.2, 0.5, 1.3, 0.8, -1.2], [0.2, -0.4, 1.8, 0.8, -1.2]]*1000

# Calculate arm trajectories
joint_trajectories = calculate_human_trajectory(target_arm_angles_complex, arm_speed) # CHANGE CHOSEN ARM MOVEMENT HERE

# - - - - - - - - - - - - - - - - - -

# Collision objects initialisation

num_joints = p.getNumJoints(robotID) # Robot total number of joints
end_effector_link_index = 8 # Defining the link index which corresponds to the end-effector
ik_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8] # Defining links for inverse kinematic use

collision_indices = [num_joints - 3]  # Links chosen to cover the end-effector (wrist and forearm)
collision_dimensions = [(0.1, 0.1)] # Capsule radius/height dimensions

# Setting collision mask for human model
collisionFilterGroup = 0
collisionFilterMask = 0
p.setCollisionFilterGroupMask(humanID, -1, collisionFilterGroup, collisionFilterMask)

capsules = []
for index, (radius, height) in zip(collision_indices, collision_dimensions):
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CAPSULE, radius=radius, height=height)
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CAPSULE, radius=radius, length=height, rgbaColor=[1, 0, 0, 0.2])

    link_state = p.getLinkState(robotID, index)
    capsule_id = p.createMultiBody(baseMass=0.000001, 
                                  baseCollisionShapeIndex=collision_shape_id, 
                                  baseVisualShapeIndex=visual_shape_id, 
                                  basePosition=link_state[4], 
                                  baseOrientation=link_state[5])
    
    enableCollision = 0
    p.setCollisionFilterPair(capsule_id, humanID, -1, -1, enableCollision)

    capsules.append((capsule_id, index))

capsules_middle = []
collision_dimensions_medium = [(0.15, 0.15)]
for index, (radius, height) in zip(collision_indices, collision_dimensions_medium):
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CAPSULE, radius=radius, height=height)
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CAPSULE, radius=radius, length=height, rgbaColor=[1, 0, 0, 0.1])

    link_state = p.getLinkState(robotID, index)
    capsule_id = p.createMultiBody(baseMass=0.000001, 
                                  baseCollisionShapeIndex=collision_shape_id, 
                                  baseVisualShapeIndex=visual_shape_id, 
                                  basePosition=link_state[4], 
                                  baseOrientation=link_state[5])
    
    enableCollision = 0
    p.setCollisionFilterPair(capsule_id, humanID, -1, -1, enableCollision)

    capsules_middle.append((capsule_id, index))

boxes = []
for index in collision_indices:
    half_extents = [0.2, 0.1, 0.2]  # Half dimensions of the box in x, y, z

    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents)
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 0.2])

    link_state = p.getLinkState(robotID, index)
    box_id = p.createMultiBody(baseMass=0.000001,
                                baseCollisionShapeIndex=collision_shape_id,
                                baseVisualShapeIndex=visual_shape_id,
                                basePosition=link_state[4],
                                baseOrientation=link_state[5]
    )

    enableCollision = 0
    p.setCollisionFilterPair(box_id, humanID, -1, -1, enableCollision)

    boxes.append((box_id, index)) 

# - - - - - - - - - - - - - - - - - - 

# Define robot target positions and max speed
target_positions = [
    [7.15041687e-13, 2.87999988e-01, 1.52600002e+00], [-0.5,  0.6,  1.3], [0.5, 0.6, 1.3], [2, 1.6, 1],
    [0, 1.6, 2.5], [2, 1.6, 1], [-2, 1.6, 1], [2, 1.6, 1.3], 
    [-2, 1.6, 1.3], [2, 1.6, 1.6], [-2, 1.6, 1.6], [2, 1.6, 1], [-2, 1.6, 2.5], 
    [2, 1.6, 2.5], [2, 1.6, 1], [0, 1.6, 2.5], [2, 1.6, 1], 
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
p.setCollisionFilterPair(robotID, humanID, -1, -1, enableCollision = 0)

# - - - - - - - - - - - - - - - - - - 

start_check = False # Flag to start recording data for graph
step_counter = 0 # Counter to keep track of the number of simulation steps
stoppage_counter = 0 # Counter to keep track of number of times robot has stopped
slowed_counter = 0
collision_counter = 0 # Counter to keep track of number of times robot has collided with human
collision_marker = False
speeds = []  # To store the speeds at each step
times = [] # To store the times at each step
prev_position = None  # To store the previous position for speed calculation

prev_half_extents = np.array([0.1, 0.1, 0.1]) * np.ones(3) # Setting initial box size parameters
repeat_trajectory = False # Flag for checking if current trajectory segment needs to be recalculated with new max_speed
target_reached = False # Flag for checking if the end-effctor has reached the target position

i = 0 # Setting the initial loop variable

# - - - - - - - - - - - - - - - - - - 

# Execute the loop
try:

    while 1:
        print("I:")
        print(i)
        if not repeat_trajectory:
            print("Next target")
            current_position = get_end_effector_position(robotID, end_effector_link_index)
            next_position = target_positions[i + 1]
            i = i + 1
        
            # Calculate the trajectory for the current segment
            joint_trajectory = calculate_trajectory_speed(current_position, next_position, max_speed, robotID, end_effector_link_index)

        repeat_trajectory = False
        target_reached = False

        # Execute the joint trajectory
        for joint_configs in joint_trajectory:
            p.setJointMotorControlArray(robotID, ik_joint_indices, p.POSITION_CONTROL, targetPositions=joint_configs)
        
            for joint_index, trajectory in zip(right_arm_joints, joint_trajectories):
                p.setJointMotorControl2(humanID, joint_index, p.POSITION_CONTROL, targetPosition=trajectory[step_counter])
            
            p.stepSimulation()
            sleep(1./240.)  # Maintain simulation rate

            step_counter += 1
            prev_speed = max_speed
            max_speed = 1

            for box_id, link_index in boxes:
                box_id, prev_half_extents = adjust_box_dynamically(robotID, link_index, box_id, prev_half_extents)
                sleep(1./240.)
                if p.getContactPoints(box_id, humanID):
                    max_speed = 0.6

            for capsule_id, link_index in capsules_middle:
                link_state = p.getLinkState(robotID, link_index)
                p.resetBasePositionAndOrientation(capsule_id, link_state[4], link_state[5])
                if p.getContactPoints(capsule_id, humanID):
                    max_speed = 1.2

            for capsule_id, link_index in capsules:
                link_state = p.getLinkState(robotID, link_index)
                p.resetBasePositionAndOrientation(capsule_id, link_state[4], link_state[5])
                if p.getContactPoints(capsule_id, humanID):
                    max_speed = 0
                    print("Stopped")
                    print(next_position)

            if step_counter >= 40:
                # Get the current position of the end effector
                current_position = get_end_effector_position(robotID, end_effector_link_index)

                # Calculate speed if it's not the first step
                if prev_position is not None:
                    time_elapsed = time.time() - start_time
                    times.append(time_elapsed)

                    speed = np.linalg.norm(current_position - prev_position) / (1./196.)  # Assuming time step of 1/240
                    speeds.append(speed)
            
                # Update the previous position
                prev_position = current_position

                if start_check == False:
                    start_time = time.time()  # Start time of the task
                    start_check = True

                if max_speed != 1:
                    slowed_counter += 1

            # Recalculate trajectory from the current position if speed changes due to collision
            if max_speed != prev_speed:
                if prev_speed == 0 and max_speed != 0:
                        stoppage_counter += 1
                print("Changing speed")
                current_position = get_end_effector_position(robotID, end_effector_link_index)
                obstacle_position = get_end_effector_position(humanID, hand_index)
                #obstacle_position = p.getClosestPoints(humanID, capsules_middle)

                if max_speed == 1.2:
                    repulsive_force = calculate_repulsive_force(current_position, obstacle_position)
                    sleep(1./240.)
                    print("Potential field activate")
                    adjusted_next_position = np.array(current_position) + repulsive_force
                    joint_trajectory_avoid = calculate_trajectory_speed(current_position, adjusted_next_position.tolist(), max_speed, robotID, end_effector_link_index)
                    joint_trajectory_append = calculate_trajectory_speed(adjusted_next_position, next_position, max_speed, robotID, end_effector_link_index)
                    joint_trajectory = joint_trajectory_avoid + joint_trajectory_append 

                else:
                    joint_trajectory = calculate_trajectory_speed(current_position, next_position, max_speed, robotID, end_effector_link_index)
        
                repeat_trajectory = True
                break

            if p.getContactPoints(robotID, humanID):
                collision_marker = True
            
            if collision_marker == True:
                if not p.getContactPoints(robotID, humanID):
                    collision_counter += 1
                    collision_marker = False

            sleep(1./240.)  

except KeyboardInterrupt:
    print("Simulation stopped manually")
finally:
    task_time = time.time() - start_time
    print(f"Total task completion time: {task_time:.2f} seconds")
    print(f"Total number of stoppages: {stoppage_counter} times")
    print(f"Total number of collisions: {collision_counter} times")

    slowed_percentage = (slowed_counter/(step_counter-40))*100
    print(f"Percentage of total time slowed: {slowed_percentage}")

    # Save data to CSV
    with open('speed_data_script4.csv', 'w', newline='') as csvfile:
        fieldnames = ['time', 'speed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for t, s in zip(times, speeds):
            writer.writerow({'time': t, 'speed': s})

    plt.figure()
    plt.plot(times, speeds, label='End-effector speed')
    plt.xlabel('Simulation time elapsed (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('End-effector speed over Simulation time')
    plt.legend()
    plt.show()

    p.disconnect()
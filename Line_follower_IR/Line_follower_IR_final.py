import time
import matplotlib.pyplot as plt
import numpy as np
import csv,os
import cv2
import keyboard
import heapq
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from scipy.ndimage import label,binary_dilation
from BAS_path_smoothening import BAS_Smoothner

# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.require('sim')

# Get handles
vision_sensors = [
    sim.getObject('/youBot/IR_R4'),
    sim.getObject('/youBot/IR_R3'),
    sim.getObject('/youBot/IR_R2'),
    sim.getObject('/youBot/IR_R1'),
    sim.getObject('/youBot/IR_M'),
    sim.getObject('/youBot/IR_L1'),
    sim.getObject('/youBot/IR_L2'),
    sim.getObject('/youBot/IR_L3'),
    sim.getObject('/youBot/IR_L4')
    
]
sensor_weights = [-4,-3,-2,-1,0,1,2,3,4]
bot_wheels = [
    sim.getObject('/youBot/rollingJoint_fl'),
    sim.getObject('/youBot/rollingJoint_rl'),
    sim.getObject('/youBot/rollingJoint_rr'),
    sim.getObject('/youBot/rollingJoint_fr')
]
def print_metrics(total_distance,total_dev,total_time,max_angle,min_angle,total_angle,max_dev,min_dev,N):
    print("Total Distance:",total_distance)
    print("Total Time:",total_time)
    print("Average Speed:", total_distance/total_time)

    print("Maximum Angle:", max_angle)
    print("Minimum Angle:", min_angle)
    print("Avg Angle:", total_angle/N)

    print("Average Deviation:", total_dev/N)
    print("Maximum Deviation:", max_dev)
    print("Minimum Deviation:", min_dev)
    
def set_movement(bot_wheels,FB_vel,LR_vel,rot):
    sim.setJointTargetVelocity(bot_wheels[0],-FB_vel-LR_vel-rot) 
    sim.setJointTargetVelocity(bot_wheels[1],-FB_vel+LR_vel-rot) 
    sim.setJointTargetVelocity(bot_wheels[2],-FB_vel-LR_vel+rot) 
    sim.setJointTargetVelocity(bot_wheels[3],-FB_vel+LR_vel+rot) 

def read_ir(sensor_handle):
    # Read the image from vision sensor
    image, [resx,resy] = sim.getVisionSensorImg(sensor_handle,1 )
    # if image is None:
    #     print("none")

    image = np.frombuffer(image, dtype=np.uint8).reshape(resy, resx, 1)
    val = np.mean(image)

    return val  # Default to white

def world_to_grid(x, y):
    """Convert world (x, y) to grid indices (i, j)."""
    j = int((origin_y - y) / cell_size)
    i = int((x + origin_x) / cell_size)
    return i, j

def grid_to_world(i, j):
    """Convert grid indices to world (x, y)."""
    x = i * cell_size - origin_x
    y = -j * cell_size + origin_y
    return x, y
def plot_lidar(points,waypoints=None,target=None):
    
    xy = points[:, :2]  # shape (N, 2)
    xy = xy[:,[1,0]]
    xy[:,0] = -xy[:,0]
    robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
    global mapping,grid
    if mapping:
        update_grid(robo_pose, xy,2)
        
    show_grid(robo_pose,grid,waypoints,target)

def get_line(start, end):

    x1, y1 = start
    x2, y2 = end
    # print("Start:", start, "End:", end)
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
        # print(points)
    if swapped:
        points.reverse()
    # print("Line points:", points)
    return points
def update_grid(robot_pose, lidar_points, occupied_radius_cells=1):
    global grid
    x_robot, y_robot, theta = robot_pose
    robot_i, robot_j = world_to_grid(x_robot, y_robot)
    theta -= np.pi / 2  # Adjust theta to match grid orientation
    for x_rel, y_rel in lidar_points:
        # Transform to world coordinates
        dist = np.sqrt(x_rel**2 + y_rel**2)
        x_world = x_robot + x_rel * np.cos(theta) - y_rel * np.sin(theta)
        y_world = y_robot + x_rel * np.sin(theta) + y_rel * np.cos(theta)
        end_i, end_j = world_to_grid(x_world, y_world)
        line_cells = get_line((robot_i, robot_j), (end_i, end_j))
        # Mark free cells
        for cell in line_cells[:-1]:
            if 0 <= cell[1] < rows and 0 <= cell[0] < cols:
                #grid first is row index, second slot is column index
                grid[cell[1], cell[0]] = 0
        # Mark a region around endpoint as occupied
        # print(dist)
        if abs(dist-5) < 0.005:
            # print('yooo')
            continue
        for di in range(-occupied_radius_cells, occupied_radius_cells + 1):
            for dj in range(-occupied_radius_cells, occupied_radius_cells + 1):
                ni, nj = end_i + di, end_j + dj
                # Optional: use a circular mask
                if di**2 + dj**2 <= occupied_radius_cells**2:
                    if 0 <= ni < rows and 0 <= nj < cols:
                        grid[nj, ni] = 1

def show_grid(robot_pose,grid,waypoints=None,target=None):
    # Map values to colors: 0=unknown(128), 1=free(255), 2=occupied(0)
    global mapping,slam
    grid = grid if mapping else grid
    img = np.full(grid.shape, 128, dtype=np.uint8)
    img[grid == 0] = 255
    img[grid == 1] = 0
    img[grid == -1] = 100
    if slam:
        img[grid == 1] = 0
        img[grid == 0] = 255
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # print("Target:", target)
    # global testing_target

    # fi,fj = world_to_grid(final_goal[0], final_goal[1])
    # i,j = world_to_grid(target[0],target[1])  
    ri, rj = world_to_grid(robot_pose[0], robot_pose[1])
    # i is the column pixel index, j is the row pixel index | i pixels to the left and j pixels down
    # if not mapping:
    #     cv2.circle(img_color, (i,j), radius=3, color=(0, 0, 255), thickness=-1) 
        # cv2.drawMarker(
        #     img_color, (fi, fj), color=(255,0,0), markerType=cv2.MARKER_TILTED_CROSS,
        #     markerSize=10, thickness=1, line_type=cv2.LINE_AA
        # )
    cv2.circle(img_color, (ri,rj), radius=5, color=(0,255, 0), thickness=-1) 
    if target is not None:
        ti,tj = world_to_grid(target[0], target[1])
        cv2.circle(img_color, (ti,tj), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.circle(img_color, (testing_target[1],testing_target[0]), radius=3, color=(0,0, 200), thickness=-1)   
    if waypoints is not None:
        for coord in waypoints:  # coords_list is your list of (x, y) tuples
            x,y = coord
            cv2.circle(img_color, (int(y), int(x)), radius=1, color=(0, 0, 255), thickness=-1)
    cv2.imshow('Occupancy Grid', img_color)
    cv2.waitKey(1)

def BAS_pid(error_l,error_r,b,D,d,d_inc,old_pid):
    # Implement PID control here
    diff = error_r - error_l
    new_d = 0.95*d +d_inc
    new_D = 0.95*D
    if (diff!= 0):
        return old_pid + np.sign(diff)*D*b , new_D, new_d
    else:
        return old_pid, D, d

def get_error(vision_sensors,sensor_weights):
    error =0
    for i in range(len(vision_sensors)):
        val = read_ir(vision_sensors[i])
        error += val*sensor_weights[i]
    return error   

def fill_closed_regions(grid):
    mask = (grid == -1)
    labeled, num_features = label(mask)
    # Find labels that touch the border
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ])))
    # Fill all regions not connected to the border
    for region in range(1, num_features + 1):
        if region not in border_labels:
            grid[labeled == region] = 1
    return grid

def inflate_obstacles(grid, thickness=2):
    # Create a mask for obstacle cells (value 2)
    obstacle_mask = (grid == 1)
    # Create a structuring element (disk or square)
    struct = np.ones((2*thickness+1, 2*thickness+1), dtype=bool)
    # Dilate the obstacle mask
    inflated = binary_dilation(obstacle_mask, structure=struct)
    # Find the new boundary (cells that are in inflated but not in original obstacle)
    boundary = inflated & ~obstacle_mask
    # Set these boundary cells to 1.5 (extended boundary)
    grid[boundary] = 1
    return grid
def point_in_obs(grid,pose):
    x, y = pose[:2]
    i, j = world_to_grid(x, y)
    return grid[i, j] == 1
    # pass

def find_target(grid,poses_on_line,robot_pose):
    # find the closest pose on the line to the robot
    min_distance = float('inf')
    closest_pose = None
    pose_idx=0
    for idx,pose in enumerate(poses_on_line):
        diff = pose[:2] - robot_pose[:2]
        distance = np.linalg.norm(diff)
        # transform pose to robot frame
        x_local = np.cos(robot_pose[2]) * diff[0] + np.sin(robot_pose[2]) * diff[1]
        y_local = -np.sin(robot_pose[2]) * diff[0] + np.cos(robot_pose[2]) * diff[1]
        pj,pi = world_to_grid(pose[0], pose[1])
        # Check if pose is valid (not out of bounds and not in obstacle
        alpha = np.arctan2(y_local, x_local)
        # print(alpha)
        if distance < min_distance and distance > 1 and not(count_ones_in_radius(grid,pi,pj,5)) and abs(alpha)<np.pi/3:
            # print("alpha", alpha)
            min_distance = distance
            closest_pose = pose
            pose_idx=idx
    return closest_pose,poses_on_line[(pose_idx+1)%len(poses_on_line)]


def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def count_ones_in_radius(grid, x, y, radius):
    rows, cols = grid.shape
    count = 0

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            # Skip center cell
            if dx == 0 and dy == 0:
                continue
            # Check bounds
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx, ny] == 1:
                    count += 1
    # print(count)
    return count


def astar(grid, start, goal):
    si,sj = world_to_grid(start[0], start[1])
    gi,gj = world_to_grid(goal[0], goal[1])
    start = (sj,si)
    goal = (gj, gi)
    # print("start:", start, "goal:", goal)
    print("start:", start, "goal:", goal)
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    new_cost = cost + np.hypot(dx, dy)  
                    priority = new_cost + heuristic(neighbor, goal) + count_ones_in_radius(grid, nx, ny,4)  # Add heuristic cost to priority queue
                    heapq.heappush(open_set, (priority, new_cost, neighbor, path + [neighbor]))
    return []


def subsample_path(path, step=3):
    return  path[::step]+[path[-1]] if path else []

def world_to_bot_frame(global_point, bot_position, bot_theta):
    dx = global_point[0] - bot_position[0]
    dy = global_point[1] - bot_position[1]
    x_local =  np.cos(-bot_theta) * dx - np.sin(-bot_theta) * dy
    y_local =  np.sin(-bot_theta) * dx + np.cos(-bot_theta) * dy
    return np.array([x_local, y_local])

def follow_till_line(waypoints,target):
    for point in waypoints:
        
        temp_goal = grid_to_world(point[1],point[0])
        while True:
            robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
            local = world_to_bot_frame(temp_goal, robot_pose[:2], robot_pose[2])
            dist = np.linalg.norm(local)
            if dist <0.05:
                break
            alpha = np.atan2(local[1],local[0]) if local[0] != 0 else np.pi/2
            set_movement(bot_wheels,3,0,- float(8*alpha**3 + 5*alpha))
            # if abs(alpha) >0.2:
            #     # rot = - np.sign(alpha)*2
            #     rot = - float(alpha**3 + 5*alpha)  
            #     set_movement(bot_wheels, 0,0, rot)  
            # else:
            #     set_movement(bot_wheels, 20*dist, 0, 0)
    print("done")
    global slam
    slam = False

    while True:
        robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")    
        delta = target[2] - robot_pose[2]
        if abs(delta) < 0.05:
            break
        set_movement(bot_wheels, 0, 0, -5*float(delta))

def ftd_pure_pursuit(waypoints, target):
    for point in waypoints:
        
        temp_goal = grid_to_world(point[1],point[0])
        while True:
            robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
            local = world_to_bot_frame(temp_goal, robot_pose[:2], robot_pose[2])
            dist = np.linalg.norm(local)
            alpha = np.atan2(local[1],local[0]) if local[0] != 0 else np.pi/2
            kappa = 2 * local[1] / (dist ** 2)
            vel = 2
    # lin speed scale = 0.06189 , ang speed scale = 0.194636
            rot = -kappa*vel 
            set_movement(bot_wheels, 0, vel, rot)
            
    print("done")
    global slam
    slam = False
    while True:
        robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")    
        delta = target[2] - robot_pose[2]
        if abs(delta) < 0.05:
            break
        set_movement(bot_wheels, 0, 0, -5*float(delta))


def get_sim_type(bas,SLAM):
    if bas and not SLAM:
        return 'BAS'
    elif not bas and SLAM:
        return 'SLAM'
    elif bas and SLAM:
        return 'BAS_SLAM'
    else:
        return 'Normal'


sim.startSimulation()
time.sleep(0.5)

Kp = 0.01
Kd=0.002
Ki = 0.0006
integral = 0
last_error = 0

path = sim.getObject('/Path')
robot = sim.getObject('/youBot')
pathData = sim.unpackDoubleTable(sim.getBufferProperty(path, 'customData.PATH'))
_,  totalLength = sim.getPathLengths(pathData, 7)

robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")

matrix = np.array(pathData, dtype=np.float64).reshape(-1, 7)
traj = matrix[:,:2]

# ============== Metrics =================================
old_robot_xy = None
total_distance = 0
start_time = time.time()
total_angle = 0
max_angle = 0
min_angle = 10000
total_dev = 0
max_dev = 0
min_dev = 1000
last_yaw = robot_pose[2]
last_pos = np.array(robot_pose[:2])
counts = 0

# ================ BAS PID ============================
pid_l = np.array([0, 0,0])
pid_r = np.array([0, 0,0])
error_l = 0
error_r = 0
d = 0.002
d_inc = 0.01*d 
#scaling factor for PIDs = D
# D = 0.0008
# D = 0.8*d*np.array([1,0.0001,0.01])
D = 0.9*d

b = np.random.randn(3)  # 3 for Kp, Ki, Kd
b = b / np.linalg.norm(b)
# start_idx = np.argmin(distances)

# ============== Mapping ============================
k = 1
grid_width = 20*k    # meters
grid_height = 20*k   # meters
cell_size = 0.05*k    # meters per cell
# Compute grid size
cols = int(grid_width / cell_size)
rows = int(grid_height / cell_size)
# Set grid origin (world coordinates of grid[0,0])
origin_x = grid_width // 2
origin_y = grid_height // 2

# ============== Flags ============================
target_speed = 4
SLAM = True # set to true if obstacles
bas = False

mapping = False
teleop = mapping
teleop_lin_vel = 0
teleop_rot_vel = 0
slam = False
grid = np.full((rows, cols),-1, dtype=np.int8) if mapping else np.load('grid3.npy')
success = "S"


if not mapping:
    grid = fill_closed_regions(grid)  # Fill closed regions if mapping is enabled
    grid = inflate_obstacles(grid, thickness=2)  # Inflate obstacles if mapping is enabled
    grid[grid==-1] = 1

# ============== Poses on the Main Line ============================    
poses_on_line = []
with open('robot_pose2.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        poses_on_line.append(np.array(row, dtype=np.float32))

# ============== Main Loop ============================

try:
    while sim.getSimulationState()!=sim.simulation_stopped:
        if teleop:
            inputs = np.array([int(keyboard.is_pressed('w')),
                               int(keyboard.is_pressed('s')),
                               int(keyboard.is_pressed('a')),
                               int(keyboard.is_pressed('d'))])
            teleop_lin_vel +=  1* (inputs[0] -inputs[1])
            teleop_rot_vel -= 0.5 * (inputs[2] - inputs[3])
            if keyboard.is_pressed('x'):
                teleop_lin_vel =0
                teleop_rot_vel = 0 
            # print(teleop_rot_vel, teleop_lin_vel)
            set_movement(bot_wheels, float(teleop_lin_vel),0,float(teleop_rot_vel))
            time.sleep(0.01)
            
        counts+=1
        if bas:
            if counts%2:
                b = np.random.randn(3)  # 3 for Kp, Ki, Kd
                b = b / np.linalg.norm(b)
                pid_l = np.array([Kp, Kd, Ki]) + d*b
                Kp,Kd,Ki = pid_l
            else:
                pid_r = np.array([Kp, Kd, Ki]) - d*b
                Kp,Kd,Ki = pid_r
        myData = sim.getBufferProperty(sim.handle_scene, "customData.lidar_points", {'noError' : True})
        points = sim.unpackTable(myData)
        points = np.array(points, dtype=np.float32).reshape(-1, 3)
        plot_lidar(points)

        if teleop:
            time.sleep(0.05)
            continue
            
        robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
        robot_angle = robot_pose[2]
        robot_xy = np.array(robot_pose[:2])  # Only x, y
        front_lidar = points[len(points)//2-30:len(points)//2+30,:2]
        rel_lidar = front_lidar 
        front_lidar_dist = np.linalg.norm(rel_lidar, axis=1)

        if np.min(front_lidar_dist) < 0.8:
            print("Obstacle detected!")
            # if slam:
            #     continue
            set_movement(bot_wheels,0,0,0)
            slam = True

            target,next_point = find_target(grid,poses_on_line,robot_pose)
            path = astar(grid, robot_xy, target[:2])
            waypoints = subsample_path(path, step=12)
            waypoints = BAS_Smoothner.smooth_path(waypoints,costmap=grid,iterations=200)
            plot_lidar(points,waypoints,target)
            follow_till_line(waypoints,target)
            
            # print(waypoints)
            
        
        distances = np.linalg.norm(traj - robot_xy, axis=1)
        deviation = np.min(distances)

        # if total_distance%3 <0.1:
        #     print(robot_pose)
        #     with open('robot_pose.csv', 'a') as f:
        #         f.write(f"{robot_pose[0]},{robot_pose[1]},{robot_pose[2]}\n")
        
        total_dev+=deviation
        max_dev = deviation if max_dev < deviation else max_dev
        min_dev = deviation if min_dev > deviation else min_dev
        diff = (robot_angle - last_yaw + np.pi) % (2 * np.pi) - np.pi
        angle_diff = abs(diff)
        total_angle+=angle_diff
        max_angle = angle_diff if angle_diff> max_angle else max_angle
        min_angle = angle_diff if angle_diff < min_angle else min_angle
        last_yaw = robot_angle

        total_distance+=np.linalg.norm(robot_xy - last_pos)
        last_pos = robot_xy

        error = get_error(vision_sensors, sensor_weights)
        integral += error
        derivative = error - last_error
        correction = Kp * error + Ki * integral + Kd * derivative
        last_error = error

        if deviation > 1:
            print("Failed")
            success = "F"
            break
        if bas:
            if counts%2:
                error_l = deviation
            else:
                error_r = deviation
                temp = BAS_pid(error_l,error_r,b,D,d,d_inc,np.array([Kp, Ki, Kd]))
                Kp,Kd,Ki = temp[0]
                d, D = temp[1:]
            
        
        set_movement(bot_wheels,target_speed,0,correction)
        time.sleep(0.001)

finally:
    set_movement(bot_wheels, 0, 0, 0)  # Stop the robot
    if not mapping:
        print_metrics(total_distance,total_dev,time.time()-start_time,max_angle,min_angle,total_angle,max_dev,min_dev,counts)
        print("Final BAS params:", Kp, Ki, Kd)
        sim_type = get_sim_type(bas,SLAM)
        # with open('simulation_metrics.csv', 'a') as f:
        #     total_time = time.time() - start_time
        #     f.write(f"{sim_type},{target_speed},{total_distance},{total_time},{total_distance/total_time},{min_angle},{max_angle},{total_angle/counts},{min_dev},{max_dev},{total_dev/counts},{success},{Kp},{Ki},{Kd}\n")
    
    time.sleep(0.5)  # Allow time for the robot to stop
    if mapping:
        np.save("grid3.npy", grid)
    sim.stopSimulation()


import time
import matplotlib.pyplot as plt
import numpy as np
import csv,os
import cv2
import keyboard
import heapq
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from scipy.ndimage import label,binary_dilation

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
def plot_lidar(points,waypoints=None):
    
    xy = points[:, :2]  # shape (N, 2)
    xy = xy[:,[1,0]]
    xy[:,0] = -xy[:,0]
    robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
    global mapping,grid
    if mapping:
        update_grid(robo_pose, xy,2)
        
    show_grid(robo_pose,grid,waypoints)

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

def show_grid(robot_pose,grid,waypoints=None):
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
    global testing_target

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
    cv2.circle(img_color, (rj,ri), radius=5, color=(0,255, 0), thickness=-1) 
    cv2.circle(img_color, (testing_target[1],testing_target[0]), radius=3, color=(0,0, 200), thickness=-1)   
    if waypoints is not None:
        for coord in waypoints:  # coords_list is your list of (x, y) tuples
            y,x = coord
            cv2.circle(img_color, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)
    cv2.imshow('Occupancy Grid', img_color)
    cv2.waitKey(1)

def BAS_pid(error_l,error_r,D,d,old_pid):
    # Implement PID control here
    diff = error_r - error_l
    new_d = 0.99*d +0.001
    new_D = 0.99*D
    if (diff!= 0):
        return old_pid + D*d*np.sign(diff) , new_D, new_d
    else:
        return old_pid, D, d


sim.startSimulation()
time.sleep(0.5)

Kp = 0.01
Kd=0.0006
Ki = 0.002
integral = 0
last_error = 0
path = sim.getObject('/Path')
robot = sim.getObject('/youBot')
old_robot_xy = None
pathData = sim.unpackDoubleTable(sim.getBufferProperty(path, 'customData.PATH'))
_,  totalLength = sim.getPathLengths(pathData, 7)
# print(totalLength)
matrix = np.array(pathData, dtype=np.float64).reshape(-1, 7)
traj = matrix[:,:2]
robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")

# ============== Metrics =================================
total_distance = 0
start_time = time.time()
total_angle = 0
max_angle = 0
min_angle = 0

total_dev = 0
max_dev = 0
min_dev = 1000

last_yaw = robot_pose[2]
last_pos = np.array(robot_pose[:2])

# print(traj)
counts = 0

# ============== plotting ============================
fig, ax = plt.subplots()
x_data = []
y_data = []
line, = ax.plot(x_data, y_data, 'r-')
# plt.ion()
# plt.show()

# ================ BAS PID ============================
pid_l = np.array([0, 0,0])
pid_r = np.array([0, 0,0])
error_l = 0
error_r = 0
d = 0.9
D = 0.99
b = np.random.randn(3)  # 3 for Kp, Ki, Kd
b = b / np.linalg.norm(b)
# start_idx = np.argmin(distances)

def get_error(vision_sensors,sensor_weights):
    error =0
    for i in range(len(vision_sensors)):
        val = read_ir(vision_sensors[i])
        error += val*sensor_weights[i]
    return error   

def fill_closed_regions(grid):
    mask = (grid == 0)
    labeled, num_features = label(mask)
    # Find labels that touch the border
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ])))
    # Fill all regions not connected to the border
    for region in range(1, num_features + 1):
        if region not in border_labels:
            grid[labeled == region] = 2
    return grid

def inflate_obstacles(grid, thickness=2):
    # Create a mask for obstacle cells (value 2)
    obstacle_mask = (grid == 2)
    # Create a structuring element (disk or square)
    struct = np.ones((2*thickness+1, 2*thickness+1), dtype=bool)
    # Dilate the obstacle mask
    inflated = binary_dilation(obstacle_mask, structure=struct)
    # Find the new boundary (cells that are in inflated but not in original obstacle)
    boundary = inflated & ~obstacle_mask
    # Set these boundary cells to 1.5 (extended boundary)
    grid[boundary] = 2
    return grid
def point_in_obs(grid,pose):
    x, y = pose[:2]
    i, j = world_to_grid(x, y)
    return grid[i, j] == 2
    # pass

def find_target(grid,poses_on_line,robot_pose):
    # find the closest pose on the line to the robot
    min_distance = float('inf')
    closest_pose = None
    for pose in poses_on_line:
        diff = pose[:2] - robot_pose[:2]
        distance = np.linalg.norm(diff)
        # transform pose to robot frame
        x_local = np.cos(robot_pose[2]) * diff[0] + np.sin(robot_pose[2]) * diff[1]
        y_local = -np.sin(robot_pose[2]) * diff[0] + np.cos(robot_pose[2]) * diff[1]

        # Check if pose is valid (not out of bounds and not in obstacle
        alpha = np.arctan2(y_local, x_local)
        # print(alpha)
        if distance < min_distance and not point_in_obs(grid, pose) and abs(alpha)<np.pi/3:
            # print("alpha", alpha)
            min_distance = distance
            closest_pose = pose
    return closest_pose



def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def count_adjacent_ones(grid, x, y):
    rows, cols = grid.shape
    count = 0

    # All 8 possible neighbors: (dx, dy)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            if grid[nx, ny] == 1:
                count += 1
    # print("count", count)
    return count

def astar(grid, start, goal):
    start = world_to_grid(start[0], start[1])
    goal = world_to_grid(goal[0], goal[1])
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
        # print("0s:", np.sum(grid == 0), "1s:", np.sum(grid == 1))
        # print("current:", current)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            
            # print("gridvalue", grid[nx,ny] if 0 <= nx < rows and 0 <= ny < cols else "out of bounds")
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    new_cost = cost + np.hypot(dx, dy)  
                    priority = new_cost + heuristic(neighbor, goal) + 10*count_adjacent_ones(grid, nx, ny)  # Add heuristic cost to priority queue
                    heapq.heappush(open_set, (priority, new_cost, neighbor, path + [neighbor]))
    return []


def subsample_path(path, step=3):
    return  path[::step]+[path[-1]] if path else []

def world_to_bot_frame(global_point, bot_position, bot_theta):
    dx = global_point[0] - bot_position[0]
    dy = global_point[1] - bot_position[1]
    # Step 2: Rotate by -theta
    x_local =  np.cos(-bot_theta) * dx - np.sin(-bot_theta) * dy
    y_local =  np.sin(-bot_theta) * dx + np.cos(-bot_theta) * dy
    return np.array([x_local, y_local])

def follow_till_line(waypoints,target):
    for point in waypoints:
        
        temp_goal = grid_to_world(point[0],point[1])
        # reached = False
        while True:
            robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
            local = world_to_bot_frame(temp_goal, robot_pose[:2], robot_pose[2])
            dist = np.linalg.norm(local)
            if dist <0.05:
                break
            alpha = np.atan2(local[1],local[0]) if local[0] != 0 else np.pi/2

            if abs(alpha) >0.2:
                rot = - 3*alpha
                set_movement(bot_wheels, 0,0, rot)  
            else:
                set_movement(bot_wheels, 20*dist, 0, 0)
    print("done")
    global slam
    slam = False
    # while True:
    #     robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")    
    #     print("target reached", target)
    #     delta = target[2] - robot_pose[2]
    #     print("delta", delta)
    #     if abs(delta) < 0.1:
    #         print("Reached line again.")
    #         break
    #     set_movement(bot_wheels, 0, 0, float(-delta))



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
mapping = False
teleop = False
teleop_lin_vel = 0
teleop_rot_vel = 0
slam = False
grid = np.full((rows, cols),-1, dtype=np.int8) if mapping else np.load('gridMap_final.npy')
testing_target = (119, 194)


if not mapping:
    grid = fill_closed_regions(grid)  # Fill closed regions if mapping is enabled
    grid = inflate_obstacles(grid, thickness=7)  # Inflate obstacles if mapping is enabled
    new_grid = np.zeros((rows, cols), dtype=np.uint8)
    new_grid[grid==0] = 1
    new_grid[grid==2] = 1
    grid = new_grid

    
poses_on_line = []
with open('robot_pose.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        poses_on_line.append(np.array(row, dtype=np.float32))

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
        # if counts%2:
        #     b = np.random.randn(3)  # 3 for Kp, Ki, Kd
        #     b = b / np.linalg.norm(b)
        #     pid_l = np.array([Kp, Kd, Ki]) + d*b
        #     Kp,Kd,Ki = pid_l
        # else:
        #     pid_r = np.array([Kp, Kd, Ki]) - d*b
        #     Kp,Kd,Ki = pid_r
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

        if np.min(front_lidar_dist < 0.99):
            print("Obstacle detected!")
            # if slam:
            #     continue
            set_movement(bot_wheels,0,0,0)
            slam = True
            # print("original 0s:", np.sum(grid == 0), "1s:", np.sum(grid == 1))
            # print(testing_target)
            # print(grid[testing_target[0], testing_target[1]])
            target = find_target(grid,poses_on_line,robot_pose)
            path = astar(grid, robot_xy, target[:2])
            waypoints = subsample_path(path, step=12)
            plot_lidar(points,waypoints)
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

        # if counts == 1:
        #     start_idx = np.argmin(distances)
        # if total_distance> 1.05*totalLength:
        #     break
        if deviation > 1:
            print("Failed")
            break
       # if counts%2:
        #     error_l = deviation
        # else:
        #     error_r = deviation
        #     temp = BAS_pid(error_l,error_r,D,d,np.array([Kp, Ki, Kd]))
        #     Kp,Kd,Ki = temp[0]
        #     d, D = temp[1:]
            
        
        set_movement(bot_wheels,6,0,correction)
        time.sleep(0.002)

finally:
    set_movement(bot_wheels, 0, 0, 0)  # Stop the robot
    if not mapping:
        print_metrics(total_distance,total_dev,time.time()-start_time,max_angle,min_angle,total_angle,max_dev,min_dev,counts)
    time.sleep(0.5)  # Allow time for the robot to stop
    if mapping:
        np.save("grid3.npy", grid)
    sim.stopSimulation()


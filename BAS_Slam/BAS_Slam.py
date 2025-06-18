import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time


def set_movement(bot_wheels,FB_vel,LR_vel,rot):
    sim.setJointTargetVelocity(bot_wheels[0],-FB_vel-LR_vel-rot) 
    sim.setJointTargetVelocity(bot_wheels[1],-FB_vel+LR_vel-rot) 
    sim.setJointTargetVelocity(bot_wheels[2],-FB_vel-LR_vel+rot) 
    sim.setJointTargetVelocity(bot_wheels[3],-FB_vel+LR_vel+rot) 

def plot_lidar(points):
    img_size = 600
    # scale = 40
    # center = img_size//2
    
    points = np.array(points, dtype=np.float32).reshape(-1, 3)
    xy = points[:, :2]  # shape (N, 2)
    # new = xy
    
    # xy = xy[:, [1, 0]]
    # xy = -xy  # Swap x and y for correct orientation
    robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
    # print(data)
    # robo_pose = sim.unpackTable(data)
    update_grid(robo_pose, xy)

    show_grid(grid)

    


    # print(xy)
    # Settings
    # print(points)

# Map lidar points to image coordinates
    # img_points = np.int32(xy * scale + center)

    # lidar_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # # Draw each point
    # for pt in img_points:
    #     cv2.circle(lidar_img, tuple(pt), 2, (0, 255, 0), -1)

    # cv2.imshow("LIDAR Points", lidar_img)

def get_line(start, end):

    x1, y1 = start
    x2, y2 = end
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
    return points

def check_dir(grid,start,end,thickness=1):
    line_cells = get_line(start, end)
    for i, j in line_cells:
        # Check a square of size (2*thickness+1) around (i, j)
        for di in range(-thickness, thickness + 1):
            for dj in range(-thickness, thickness + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    if grid[ni, nj] == 2:
                        return True
    return False

k = 1
# Parameters
grid_width = 20*k    # meters
grid_height = 20*k   # meters
cell_size = 0.05*k    # meters per cell

# Compute grid size
cols = int(grid_width / cell_size)
rows = int(grid_height / cell_size)

# Initialize grid: 0 = unknown, 1 = free, 2 = occupied
grid = np.zeros((rows, cols), dtype=np.uint8)

# Set grid origin (world coordinates of grid[0,0])
origin_x = grid_width // 2
origin_y = grid_height // 2

def world_to_grid(x, y):
    """Convert world (x, y) to grid indices (i, j)."""
    i = int(-(y - origin_y) / cell_size)
    j = int((x + origin_x) / cell_size)
    return i, j

def grid_to_world(i, j):
    """Convert grid indices to world (x, y)."""
    x = j * cell_size + origin_x
    y = i * cell_size + origin_y
    return x, y

def update_grid(robot_pose, lidar_points, occupied_radius_cells=1):
    x_robot, y_robot, theta = robot_pose
    robot_i, robot_j = world_to_grid(x_robot, y_robot)
    for x_rel, y_rel in lidar_points:
        # Transform to world coordinates
        x_world = x_robot + x_rel * np.cos(theta) - y_rel * np.sin(theta)
        y_world = y_robot + x_rel * np.sin(theta) + y_rel * np.cos(theta)
        end_i, end_j = world_to_grid(x_world, y_world)
        line_cells = get_line((robot_i, robot_j), (end_i, end_j))
        # Mark free cells
        for cell in line_cells[:-1]:
            if 0 <= cell[0] < rows and 0 <= cell[1] < cols:
                grid[cell[0], cell[1]] = 1
        # Mark a region around endpoint as occupied
        for di in range(-occupied_radius_cells, occupied_radius_cells + 1):
            for dj in range(-occupied_radius_cells, occupied_radius_cells + 1):
                ni, nj = end_i + di, end_j + dj
                # Optional: use a circular mask
                if di**2 + dj**2 <= occupied_radius_cells**2:
                    if 0 <= ni < rows and 0 <= nj < cols:
                        grid[ni, nj] = 2

def show_grid(grid):
    # Map values to colors: 0=unknown(128), 1=free(255), 2=occupied(0)
    img = np.full(grid.shape, 128, dtype=np.uint8)
    img[grid == 1] = 255
    img[grid == 2] = 0
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # x,y = target
    j,i = world_to_grid(target[1],target[0])  # Robot position in grid
    cv2.circle(img_color, (j, i), radius=3, color=(0, 0, 255), thickness=-1)  # Filled red dot

    cv2.imshow('Occupancy Grid', img_color)
    cv2.waitKey(1)
 
# def

def pose_cost(grid, pose,target, radius, value_map):
    
    rows, cols = grid.shape
    i0, j0 = pose[:2]
    i0, j0 = world_to_grid(i0, j0)
    values = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = i0 + di, j0 + dj
            # print("ni,nj:",ni,nj)
            if 0 <= ni < rows and 0 <= nj < cols:
                # Optional: use circular mask
                # if di**2 + dj**2 <= radius**2:
                cell_value = grid[ni, nj]
                # print("cell_value:",cell_value)
                values.append(value_map.get(cell_value, 0))
    start = world_to_grid(pose[0], pose[1])
    end = world_to_grid(target[0], target[1])

    if check_dir(grid,start,end):
        print("Direction blocked")
        return 999
    elif values:
        # print("Values:", values)
        return sum(values) / len(values)
    else:
        return 1  # High cost if no valid cells



def BAS_target(grid,pose,d,D):
    radius = 2
    value_map = {0: 0, 1: 1, 2: 999}  # Unknown=1, Free=0, Occupied=999
    angle = np.random.uniform(0, 2 * np.pi)
    b = np.array([np.cos(angle), np.sin(angle)])
    p_l = pose - d*b
    p_r = pose + d*b
    diff = pose_cost(grid,pose,p_l,radius,value_map)-pose_cost(grid,pose,p_r,radius,value_map)
    if diff == 0:
        target = pose
    else :
        target = pose + D*b*(diff/abs(diff))
    
    print("target:",target)
    return target





# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.require('sim')

# Get handles
vision_sensor_handle = sim.getObject('/youBot/visionSensor')
bot_wheels = [
    sim.getObject('/youBot/rollingJoint_fl'),
    sim.getObject('/youBot/rollingJoint_rl'),
    sim.getObject('/youBot/rollingJoint_rr'),
    sim.getObject('/youBot/rollingJoint_fr')
]

def line_follower(vision_sensor_handle, bot_wheels):
    img, [resX, resY] = sim.getVisionSensorImg(vision_sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    # Convert to grayscale and threshold for black line
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)  # Black=white in binary

    # Calculate centroid of the black area (line)
    M = cv2.moments(binary)
    cx = int(M['m10'] / M['m00'])
        
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # Draw centroid for visualization
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        # Calculate error from image center
        error = cx - resX // 2
    else:
        error = 0  # No line detected, go straight or stop


    # Simple proportional controller for steering
    Kp = 0.05
    base_speed = 8
    # Send wheel speeds
    set_movement(bot_wheels,base_speed/(1+0.05*abs(error)),0,Kp*error)
    # Show vision sensor image (optional)
    cv2.imshow('Vision Sensor', img)
# Start simulation
target = np.array([0, 0])  # Initial target position
sim.startSimulation()
time.sleep(0.5)  # Let the simulation initialize

def nav_to_target(target):
    # robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
    # target = BAS_target(grid,robo_pose[:2],2,0.2)
    print("Target:",target)
    # Move towards target
    while True:
        robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
        if np.linalg.norm(robo_pose[:2] - target) < 0.1:  # Close enough to target
            break
        # Calculate movement direction and speed
        direction = target - robo_pose[:2]
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction /= distance  # Normalize
        set_movement(bot_wheels, 5 * direction[1], 5 * direction[0], 0)  # Adjust speeds as needed
        # time.sleep(0.1)  # Adjust loop timing as needed

def world_to_bot_frame(global_point, bot_position, bot_theta):
    dx = global_point[0] - bot_position[0]
    dy = global_point[1] - bot_position[1]
    # Step 2: Rotate by -theta
    x_local =  np.cos(-bot_theta) * dx - np.sin(-bot_theta) * dy
    y_local =  np.sin(-bot_theta) * dx + np.cos(-bot_theta) * dy
    return np.array([x_local, y_local])

try:
    while sim.getSimulationState() != sim.simulation_stopped:
        # Get image from vision sensor
        # line_follower(vision_sensor_handle, bot_wheels)
        myData = sim.getBufferProperty(sim.handle_scene, "customData.lidar_points", {'noError' : True})
        # myData = sim.getFloatArrayProperty(sim.handle_scene, "lidar", dict options = {})
        points = sim.unpackTable(myData)
        
        plot_lidar(points)    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
        print(robo_pose[2])
        target = BAS_target(grid,robo_pose[:2],2,0.2)
        time.sleep(1)  # Adjust loop timing as needed

        
finally:
    sim.stopSimulation()
    cv2.destroyAllWindows()
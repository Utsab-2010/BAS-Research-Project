import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
from scipy.ndimage import label,binary_dilation


def set_movement(bot_wheels,FB_vel,LR_vel,rot):
    sim.setJointTargetVelocity(bot_wheels[0],-FB_vel-LR_vel-rot) 
    sim.setJointTargetVelocity(bot_wheels[1],-FB_vel+LR_vel-rot) 
    sim.setJointTargetVelocity(bot_wheels[2],-FB_vel-LR_vel+rot) 
    sim.setJointTargetVelocity(bot_wheels[3],-FB_vel+LR_vel+rot) 

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

def plot_lidar(points):
    points = np.array(points, dtype=np.float32).reshape(-1, 3)
    xy = points[:, :2]  # shape (N, 2)
    xy = xy[:,[1,0]]
    xy[:,0] = -xy[:,0]
    robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
    # print(data)
    # robo_pose = sim.unpackTable(data)
    update_grid(robo_pose, xy)
    show_grid(robo_pose,grid)

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

def check_dir(grid,start,end,thickness=3):
    line_cells = get_line(start, end)
    for i, j in line_cells:
        # Check a square of size (2*thickness+1) around (i, j)
        for di in range(-thickness, thickness + 1):
            for dj in range(-thickness, thickness + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    if grid[nj, ni] == 2:
                        return True
    return False

def world_to_grid(x, y):
    """Convert world (x, y) to grid indices (i, j)."""
    j = int((origin_y - y) / cell_size)
    i = int((x + origin_x) / cell_size)
    return i, j

def grid_to_world(i, j):
    """Convert grid indices to world (x, y)."""
    x = j * cell_size - origin_x
    y = -i * cell_size + origin_y
    return x, y

def update_grid(robot_pose, lidar_points, occupied_radius_cells=1):
    global dyna_grid
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
                dyna_grid[cell[1], cell[0]] = 1
        # Mark a region around endpoint as occupied
        if abs(dist-5) < 0.05:
            continue
        for di in range(-occupied_radius_cells, occupied_radius_cells + 1):
            for dj in range(-occupied_radius_cells, occupied_radius_cells + 1):
                ni, nj = end_i + di, end_j + dj
                # Optional: use a circular mask
                if di**2 + dj**2 <= occupied_radius_cells**2:
                    if 0 <= ni < rows and 0 <= nj < cols:
                        dyna_grid[nj, ni] = 2

def show_grid(robot_pose,grid):
    # Map values to colors: 0=unknown(128), 1=free(255), 2=occupied(0)
    img = np.full(grid.shape, 128, dtype=np.uint8)
    
    img[grid == 1] = 255
    img[grid == 2] = 0
    img[grid == 3] = 100
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print("Target:", target)
    global final_goal
    fi,fj = world_to_grid(final_goal[0], final_goal[1])
    i,j = world_to_grid(target[0],target[1])  
    ri, rj = world_to_grid(robot_pose[0], robot_pose[1])
    # i is the column pixel index, j is the row pixel index | i pixels to the left and j pixels down
    cv2.circle(img_color, (i,j), radius=3, color=(0, 0, 255), thickness=-1)  # Filled red dot for traget
    
    cv2.drawMarker(
        img_color, (fi, fj), color=(255,0,0), markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=10, thickness=1, line_type=cv2.LINE_AA
    )
    cv2.circle(img_color, (ri,rj), radius=5, color=(0,255, 0), thickness=-1)  # Filled blue dot for final goal

    cv2.imshow('Occupancy Grid', img_color)
    cv2.waitKey(1)
 
def pose_cost(grid, pose,target,goal,  radius=0, value_map={0: 0, 1: 1, 2: 999,3: 999}):
    global last_dist_cost
    # rows, cols = grid.shape
    # i0, j0 = pose[:2]
    # i0, j0 = world_to_grid(i0, j0)
    # values = []
    # for di in range(-radius, radius + 1):
    #     for dj in range(-radius, radius + 1):
    #         ni, nj = i0 + di, j0 + dj
    #         # print("ni,nj:",ni,nj)
    #         if 0 <= ni < rows and 0 <= nj < cols:
    #             # Optional: use circular mask
    #             # if di**2 + dj**2 <= radius**2:
    #             cell_value = grid[nj, ni]
    #             # print("cell_value:",cell_value)
    #             values.append(value_map.get(cell_value, 0))
    tx,ty = target[:2]
    tx,ty = world_to_grid(tx, ty)
    value = grid[ty, tx]
    global dyna_grid
    start = world_to_grid(pose[0], pose[1])
    end = world_to_grid(target[0], target[1])

    if check_dir(dyna_grid,start,end):
        print("Direction blocked")
        return 999
    elif value in (0,2,3):
        # print("Values:", values)
        return 999  # High cost for occupied cells
    else:
        temp = float(np.sum((target[:2] - goal[:2])**2))
        if temp < last_dist_cost+0.3:
            last_dist_cost = temp
            return temp
        else:
            return 999

def BAS_target(grid,pose,goal):
    global d, D
    radius = 2
    value_map = {0: 0, 1: 1, 2: 999}  # Unknown=1, Free=0, Occupied=999
    angle = np.random.uniform(0, 2 * np.pi)
    b = np.array([np.cos(angle), np.sin(angle)])
    # print("dir", b)
    p_l = pose - d*b
    p_r = pose + d*b
    diff = pose_cost(grid,pose,p_l,goal,radius,value_map)-pose_cost(grid,pose,p_r,goal,radius,value_map)
    if diff == 0:
        target = np.array([0, 0])  # No direction preference
    else :
        target = pose + D*b*(diff/abs(diff))
        
    d = 0.95*d + 0.01
    D = 0.95*D 
    # print("target:",target)
    return target

def nav_to_target(target):
    global path_length, cumm_angle 
    # Move towards target
    while True:
        start_time = time.time()
        myData = sim.getBufferProperty(sim.handle_scene, "customData.lidar_points", {'noError' : True})
        # myData = sim.getFloatArrayProperty(sim.handle_scene, "lidar", dict options = {})
        points = sim.unpackTable(myData)
        plot_lidar(points) 
        # plot_lidar(points)
        robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
        local = world_to_bot_frame(target, robo_pose[:2], robo_pose[2])
        dist = np.linalg.norm(local)
        alpha = math.atan2(local[1],local[0]) if local[0] != 0 else np.pi/2
        curvature = 2*local[1]/(local[0]**2 + local[1]**2) if local[0] != 0 else 0
        if dist < 0.3:
            print("Reached target")
            break
        angular_velocity = 10 * curvature  # Adjust this factor as needed
        if abs(alpha) >0.3:
            rot = -1*(alpha)*abs(alpha) - alpha
            set_movement(bot_wheels, 0,0, rot)  
            dt = time.time() - start_time
            cumm_angle += abs(dt*rot*0.05/0.081)*3.14/180
        # set_movement(bot_wheels,10,0, -angular_velocity*2)
        else:
            set_movement(bot_wheels, 5*dist, 0, 0)
            dt = time.time() - start_time
            path_length += abs(dt*5*dist*0.05)*3.14/180
            time.sleep(0.5)
        # time.sleep(0.1)  # Adjust loop timing as needed

def world_to_bot_frame(global_point, bot_position, bot_theta):
    dx = global_point[0] - bot_position[0]
    dy = global_point[1] - bot_position[1]
    # Step 2: Rotate by -theta
    x_local =  np.cos(-bot_theta) * dx - np.sin(-bot_theta) * dy
    y_local =  np.sin(-bot_theta) * dx + np.cos(-bot_theta) * dy
    return np.array([x_local, y_local])

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
    grid[boundary] = 3
    return grid


# =========================Main Code Starts Here=========================
mapping = False
# Envrionment setup
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

# Initialize grid/ Load grid from file if mapping is disabled
grid = np.zeros((rows, cols), dtype=np.uint8) if mapping else np.load("gridMap.npy")
if not mapping:
    grid = fill_closed_regions(grid)  # Fill closed regions if mapping is enabled
    grid = inflate_obstacles(grid, thickness=4)  # Inflate obstacles if mapping is enabled
dyna_grid = np.zeros((rows, cols), dtype=np.uint8)  # Dynamic grid for navigation

last_dist_cost = 9999

# Navigation parameters
d = 3 # initial antenna size
D = 2 # actual lookahead distance

# Navigation Metrics
path_length = 0.0
time_taken = 0.0
cumm_angle = 0.0

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

final_goal = np.array([2, 1.5])  # Final goal position 
target = np.array([0, 0])  # Initial target position
sim.startSimulation()
time.sleep(0.5)  # Let the simulation initialize
start_time = time.time()

try:
    while sim.getSimulationState() != sim.simulation_stopped:
        # line_follower(vision_sensor_handle, bot_wheels)
        myData = sim.getBufferProperty(sim.handle_scene, "customData.lidar_points", {'noError' : True})
        points = sim.unpackTable(myData)
        robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")

        plot_lidar(points) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if np.linalg.norm(robo_pose[:2] - final_goal) < 0.5:
            print("--------------Reached final goal--------------")
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time taken: {time_taken:.2f} seconds")
            print(f"Path length: {path_length:.2f} meters")
            print(f"Cumulative angle turned: {cumm_angle:.2f} radians")
            print(f"Final pose: {robo_pose}")
            print("--------------Simulation Ended--------------")
            break

        target = BAS_target(grid,robo_pose[:2],final_goal)
        print("new target iteration")

        if target.any() :
            nav_to_target(target)
        else:
            set_movement(bot_wheels, 0, 0, 0)  # Stop the robot if no target 
        time.sleep(0.1)  # Adjust loop timing as needed

finally:
    set_movement(bot_wheels, 0, 0, 0)  # Stop the robot
    time.sleep(0.5)  # Allow time for the robot to stop
    sim.stopSimulation()
    cv2.destroyAllWindows()
    if mapping:
        np.save("gridMap2.npy", grid)


# TODO: Add mapping/nav mode as user input
# TODO: Create the entire nav mode - DONE
# TODO: Add safety around obstacles and fill obstacles in the grid - DONE
# TODO: add dynamic antenna size - DONE
# TODO: Organize code into differnt files
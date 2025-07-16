import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
from scipy.ndimage import label,binary_dilation
import matplotlib.pyplot as plt
import csv,os


# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.require('sim')

# Get handles
vision_sensors = [
    sim.getObject('/youBot/right_ir'),
    sim.getObject('/youBot/mid_ir'),
    sim.getObject('/youBot/left_ir')
]
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

# Add this function after the existing function definitions
def append_metrics_to_csv(filename, total_distance, total_dev, total_time, max_angle, min_angle, total_angle, max_dev, min_dev, counts, final_kp, final_ki, final_kd):
    """Append metrics to CSV file. Creates file with headers if it doesn't exist."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Total_Distance', 'Total_Time', 'Average_Speed', 'Max_Angle', 'Min_Angle', 
                     'Avg_Angle', 'Average_Deviation', 'Max_Deviation', 'Min_Deviation', 
                     'Total_Deviation', 'Loop_Count', 'Final_Kp', 'Final_Ki', 'Final_Kd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Calculate derived metrics
        avg_speed = total_distance / total_time if total_time > 0 else 0
        avg_angle = total_angle / counts if counts > 0 else 0
        avg_deviation = total_dev / counts if counts > 0 else 0
        
        # Write the metrics row
        writer.writerow({
            'Total_Distance': total_distance,
            'Total_Time': total_time,
            'Average_Speed': avg_speed,
            'Max_Angle': max_angle,
            'Min_Angle': min_angle,
            'Avg_Angle': avg_angle,
            'Max_Deviation': max_dev,
            'Min_Deviation': min_dev,
            'Average_Deviation': avg_deviation,
            'Total_Deviation': total_dev,
            'Final_Kp': final_kp,
            'Final_Ki': final_ki,
            'Final_Kd': final_kd
        })

def world_to_grid(x, y):
    """Convert world (x, y) to grid indices (i, j)."""
    j = int((origin_y - y) / cell_size)
    i = int((x + origin_x) / cell_size)
    return i, j
def plot_lidar(points):
    points = np.array(points, dtype=np.float32).reshape(-1, 3)
    xy = points[:, :2]  # shape (N, 2)
    xy = xy[:,[1,0]]
    xy[:,0] = -xy[:,0]
    robo_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")

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
                grid[cell[1], cell[0]] = 1
        # Mark a region around endpoint as occupied
        if abs(dist-5) < 0.05:
            continue
        for di in range(-occupied_radius_cells, occupied_radius_cells + 1):
            for dj in range(-occupied_radius_cells, occupied_radius_cells + 1):
                ni, nj = end_i + di, end_j + dj
                # Optional: use a circular mask
                if di**2 + dj**2 <= occupied_radius_cells**2:
                    if 0 <= ni < rows and 0 <= nj < cols:
                        grid[nj, ni] = 2

def show_grid(robot_pose,grid):
    # Map values to colors: 0=unknown(128), 1=free(255), 2=occupied(0)
    global mapping
    grid = grid if mapping else grid
    img = np.full(grid.shape, 128, dtype=np.uint8)
    img[grid == 1] = 255
    img[grid == 2] = 0
    img[grid == 3] = 100
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # print("Target:", target)
    global target

    # fi,fj = world_to_grid(final_goal[0], final_goal[1])
    i,j = world_to_grid(target[0],target[1])  
    ri, rj = world_to_grid(robot_pose[0], robot_pose[1])
    # i is the column pixel index, j is the row pixel index | i pixels to the left and j pixels down
    if not mapping:
        cv2.circle(img_color, (i,j), radius=3, color=(0, 0, 255), thickness=-1) 
        # cv2.drawMarker(
        #     img_color, (fi, fj), color=(255,0,0), markerType=cv2.MARKER_TILTED_CROSS,
        #     markerSize=10, thickness=1, line_type=cv2.LINE_AA
        # )
    cv2.circle(img_color, (ri,rj), radius=5, color=(0,255, 0), thickness=-1)  

    cv2.imshow('Occupancy Grid', img_color)
    cv2.waitKey(1)

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




def BAS_pid(error_l,error_r,D,d,old_pid):
    # Implement PID control here
    diff = error_r - error_l
    new_d = 0.99*d +0.001
    new_D = 0.99*D
    if (diff!= 0):
        return old_pid + D*d*np.sign(diff) , new_D, new_d
    else:
        return old_pid, D, d

mapping = 0
sim.startSimulation()
time.sleep(0.5)

Kp = 18
Kd=1
Ki = 8
integral = 0
last_error = 0
path = sim.getObject('/Path')
robot = sim.getObject('/youBot')
old_robot_xy = None
pathData = sim.unpackDoubleTable(sim.getBufferProperty(path, 'customData.PATH'))
matrix = np.array(pathData, dtype=np.float64).reshape(-1, 7)
traj = matrix[:,:2]
robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
target_handle = sim.getObject('/Target')

# ============== Metrics =================================
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
d = 0.99
D = 0.9
b = np.random.randn(3)  # 3 for Kp, Ki, Kd
b = b / np.linalg.norm(b)

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
grid = np.zeros((rows, cols), dtype=np.uint8) 

start_idx = None
measurement_no = 0
use_bas = True
try:
    while sim.getSimulationState()!=sim.simulation_stopped:
        counts+=1
        myData = sim.getBufferProperty(sim.handle_scene, "customData.lidar_points", {'noError' : True})
        points = sim.unpackTable(myData)
        target = sim.getObjectPosition(target_handle,-1)
        if mapping:
            plot_lidar(points)

        # print(target)
        if use_bas:
            if counts%2:
                b = np.random.randn(3)  # 3 for Kp, Ki, Kd
                b = b / np.linalg.norm(b)
                pid_l = np.array([Kp, Kd, Ki]) + d*b
                Kp,Kd,Ki = pid_l
            else:
                pid_r = np.array([Kp, Kd, Ki]) - d*b
                Kp,Kd,Ki = pid_r
            
        right_val = read_ir(vision_sensors[0])
        mid_val   = read_ir(vision_sensors[1])
        left_val  = read_ir(vision_sensors[2])

        # robot_pos = sim.getObjectPosition(robot, -1)
        robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")


        robot_angle = robot_pose[2]
        robot_xy = np.array(robot_pose[:2])  # Only x, y
        dx = target[0] - robot_xy[0]
        dy = target[1] - robot_xy[1]
        local_x = dx * np.cos(robot_angle) + dy * np.sin(robot_angle)
        local_y = -dx * np.sin(robot_angle) + dy * np.cos(robot_angle)

        alpha = np.arctan2(local_y,local_x)
        # print("Angle:",alpha)
        error = -alpha 

        distances = np.linalg.norm(traj - robot_xy, axis=1)
        deviation = np.min(distances)
        if counts == 1:
            start_idx = np.argmin(distances)
        elif np.argmin(distances) == start_idx and counts > 20:
            print("Loop completed!")
            # Calculate final time and append metrics to CSV
            final_time = time.time() - start_time
            csv_filename = "line_follower_metrics.csv"  # You can make this user-configurable
            append_metrics_to_csv(csv_filename, total_distance, total_dev, final_time, 
                                max_angle, min_angle, total_angle, max_dev, min_dev, 
                                counts, Kp, Ki, Kd)
            # print(f"Metrics appended to {csv_filename}")
            total_distance = 0
            start_time = time.time()
            total_angle = 0
            max_angle = 0
            min_angle = 10000

            total_dev = 0
            max_dev = 0
            min_dev = 1000
            counts = 0
            if measurement_no >=10:
                break
            else:
                measurement_no+=1
                continue
        
        
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

        integral += error
        derivative = error - last_error
        correction = Kp * error 
        last_error = error
        if use_bas:
            if counts%2:
                error_l = deviation
            else:
                error_r = deviation
                temp = BAS_pid(error_l,error_r,D,d,np.array([Kp, Ki, Kd]))
                Kp,Kd,Ki = temp[0]
                d, D = temp[1:]
            
        set_movement(bot_wheels,16,0,correction)
        time.sleep(0.01)

finally:
    # lin speed scale = 0.06189 , ang speed scale = 0.194636
    set_movement(bot_wheels, 0, 0, 0)  # Stop the robot
    print_metrics(total_distance,total_dev,time.time()-start_time,max_angle,min_angle,total_angle,max_dev,min_dev,counts)
    print("Final BAS params:", Kp, Ki, Kd)
    time.sleep(0.5)  # Allow time for the robot to stop
    sim.stopSimulation()


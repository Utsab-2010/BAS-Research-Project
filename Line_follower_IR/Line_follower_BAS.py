import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
from scipy.ndimage import label,binary_dilation
import matplotlib.pyplot as plt



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


sim.startSimulation()
time.sleep(0.5)

Kp = 5
Kd=1
Ki = 1
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
d = 0.8
D = 0.7
b = np.random.randn(3)  # 3 for Kp, Ki, Kd
b = b / np.linalg.norm(b)





try:
    while sim.getSimulationState()!=sim.simulation_stopped:
        counts+=1

        target = sim.getObjectPosition(target_handle,-1)
        print(target)
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

        total_dev+=deviation
        max_dev = deviation if max_dev < deviation else max_dev
        min_dev = deviation if min_dev > deviation else min_dev

        angle_diff = abs(robot_angle - last_yaw)
        total_angle+=angle_diff
        max_angle = angle_diff if angle_diff> max_angle else max_angle
        min_angle = angle_diff if angle_diff < min_angle else min_angle
        last_yaw = robot_angle

        total_distance+=np.linalg.norm(robot_xy - last_pos)
        last_pos = robot_xy

        # error = left_val - right_val
        integral += error
        derivative = error - last_error
        correction = Kp * error 
        last_error = error

        if counts%2:
            error_l = deviation
        else:
            error_r = deviation
            temp = BAS_pid(error_l,error_r,D,d,np.array([Kp, Ki, Kd]))
            Kp,Kd,Ki = temp[0]
            d, D = temp[1:]
            
        # print("Error:", correction)
        set_movement(bot_wheels,10,0,correction)
        time.sleep(0.01)

finally:
    set_movement(bot_wheels, 0, 0, 0)  # Stop the robot
    print_metrics(total_distance,total_dev,time.time()-start_time,max_angle,min_angle,total_angle,max_dev,min_dev,counts)
    print("Final BAS params:", Kp, Ki, Kd)
    time.sleep(0.5)  # Allow time for the robot to stop
    sim.stopSimulation()


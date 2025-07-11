import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
from scipy.ndimage import label,binary_dilation



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




# def BAS_pid()/


sim.startSimulation()
time.sleep(0.5)

Kp = 0.12
Kd=0.01
Ki = 0.001
integral = 0
last_error = 0
path = sim.getObject('/Path')
robot = sim.getObject('/youBot')
old_robot_xy = None
pathData = sim.unpackDoubleTable(sim.getBufferProperty(path, 'customData.PATH'))
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

print(traj)
counts = 0

# ================ BAS PID ============================
pid_l = np.array([0, 0,0])
pid_r = np.array([0, 0,0])


try:
    while sim.getSimulationState()!=sim.simulation_stopped:
        counts+=1
        # if counts%2:
        #     Kp,Kd,Ki = pid_l
        # else:
        #     Kp,Kd,Ki = pid_r
            
        right_val = read_ir(vision_sensors[0])
        mid_val   = read_ir(vision_sensors[1])
        left_val  = read_ir(vision_sensors[2])

        # robot_pos = sim.getObjectPosition(robot, -1)
        robot_pose = sim.getFloatArrayProperty(sim.handle_scene, "signal.robo_pose")
        robot_angle = robot_pose[2]
        robot_xy = np.array(robot_pose[:2])  # Only x, y

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

        error = left_val - right_val
        integral += error
        derivative = error - last_error
        correction = Kp * error + Ki * integral + Kd * derivative
        last_error = error

        
        
        set_movement(bot_wheels,6,0,correction)
        time.sleep(0.01)

finally:
    set_movement(bot_wheels, 0, 0, 0)  # Stop the robot
    print_metrics(total_distance,total_dev,time.time()-start_time,max_angle,min_angle,total_angle,max_dev,min_dev,counts)
    time.sleep(0.5)  # Allow time for the robot to stop
    sim.stopSimulation()


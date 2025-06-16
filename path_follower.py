import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time


def set_movement(bot_wheels,FB_vel,LR_vel,rot):
    sim.setJointTargetVelocity(bot_wheels[0],-FB_vel-LR_vel-rot) 
    sim.setJointTargetVelocity(bot_wheels[1],-FB_vel+LR_vel-rot) 
    sim.setJointTargetVelocity(bot_wheels[2],-FB_vel-LR_vel+rot) 
    sim.setJointTargetVelocity(bot_wheels[3],-FB_vel+LR_vel+rot) 

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


# Start simulation
sim.startSimulation()
time.sleep(0.5)  # Let the simulation initialize

try:
    while sim.getSimulationState() != sim.simulation_stopped:
        # Get image from vision sensor
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
        Kp = 0.12
        base_speed = 5
        # Send wheel speeds
        set_movement(bot_wheels,base_speed,0,Kp*error)
        # Show vision sensor image (optional)
        cv2.imshow('Vision Sensor', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)  # Adjust loop timing as needed

        
finally:
    sim.stopSimulation()
    cv2.destroyAllWindows()
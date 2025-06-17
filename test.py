from time import sleep
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import numpy as np
import cv2
client = RemoteAPIClient()
sim = client.getObject('sim')
# sensor1Handle = sim.getObject('/VisionSensor')
# sensor2Handle = sim.getObject('/PassiveVisionSensor')
# Read the signal

    # scan_data is now a tuple of distances or 3D points
# dataString = sim.getFloatSignal('lidar_scan_data')

# points = np.array(lidar_points_1d).reshape(-1, 3)

# plt.ion()  # Interactive mode on
# fig, ax = plt.subplots()
# # scatter = ax.scatter(points[:,0], points[:,1])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('LIDAR Points')
# ax.grid(True)
# # ax.set_xlim(-0.5,0.5)
# # ax.set_ylim(--0.5,0.5)
# scatter = ax.scatter([], [])

# def update_lidar_points(new_points_1d):
#     new_points = np.array(new_points_1d).reshape(-1, 3)
#     print("newPOints:",new_points)
#     scatter.set_offsets(new_points[:, :2])
#     ax.set_xlim(min(new_points[:,0]) - 1, max(new_points[:,0]) + 1)
#     ax.set_ylim(min(new_points[:,1]) - 1, max(new_points[:,1]) + 1)
#     fig.canvas.draw_idle()
#     plt.pause(0.01)

img_size = 600
scale = 50  # pixels per meter, adjust as needed
center = img_size // 2
sim.startSimulation()
while sim.getSimulationTime() < 50:
    # myData = sim.getFloatProperty(sim.handle_scene, "signal.myFloatSignal", {'noError' : True})
    myData = sim.getBufferProperty(sim.handle_scene, "customData.myTag", {'noError' : True})
# image, resolution = sim.getVisionSensorImg(sensor1Handle)
# if myData:
# # Unpack the float table
#     # scan_data = struct.unpack('f' * (len(dataString)//4), dataString)
#     print(sim.unpackTable(myData))
    points = sim.unpackTable(myData)
    points = np.array(points, dtype=np.float32).reshape(-1, 3)
    xy = points[:, :2]  # shape (N, 2)
    # Settings
    # Map lidar points to image coordinates
    img_points = np.int32(xy * scale + center)
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Draw each point
    for pt in img_points:
        cv2.circle(img, tuple(pt), 2, (0, 255, 0), -1)

    cv2.imshow("LIDAR Points", img)
    cv2.waitKey(0)

# print(points)
# update_lidar_points(points)
# print(max(points))
# sim.setVisionSensorImg(sensor2Handle, image)

# for x, y in zip(x_values, y_values):
# sim.setGraphUserData(graph_handle, 'x', x)
# sim.setGraphUserData(graph_handle, 'y', y)
# sim.handleGraph(graph_handle, sim.getSimulationTime()) 

cv2.destroyAllWindows()
print('hi')
sleep(1)
sim.stopSimulation()
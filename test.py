from time import sleep
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
import numpy as np
import cv2
client = RemoteAPIClient()
sim = client.getObject('sim')


img_size = 600
scale = 50  # pixels per meter, adjust as needed
center = img_size // 2
sim.startSimulation()
try:
    plt.ion()
    fig, ax = plt.subplots()

    while sim.getSimulationState() != sim.simulation_stopped:
        # myData = sim.getFloatProperty(sim.handle_scene, "signal.myFloatSignal", {'noError' : True})
        myData = sim.getBufferProperty(sim.handle_scene, "customData.lidar_points", {'noError' : True})
    # image, resolution = sim.getVisionSensorImg(sensor1Handle)
    # if myData:
    # # Unpack the float table
    #     # scan_data = struct.unpack('f' * (len(dataString)//4), dataString)
    #     print(sim.unpackTable(myData))
        points = sim.unpackTable(myData)
        points = np.array(points, dtype=np.float32).reshape(-1, 3)
        xy = points[:, :2]  # shape (N, 2)
        xy = xy[:,[1,0]]
        xy[:,0] = -xy[:,0]  # Invert x-axis 
        print(xy)
        ax.clear()
        ax.scatter(xy[:, 0], xy[:, 1], c='b', marker='o')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        plt.draw()
        plt.pause(0.2)
        # Map lidar points to image coordinates
        # img_points = np.int32(xy * scale + center)
        # img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # # Draw each point
        # for pt in img_points:
        #     # print(pt)
        #     cv2.circle(img, tuple(pt), 2, (0, 255, 0), -1)
        # print("hmm")
        # cv2.imshow("LIDAR Points", img)
        # cv2.waitKey(1)
        # sleep(0.2)
        print("testing")
        plt.show()
    # print(points)
    # update_lidar_points(points)
    # print(max(points))
    # sim.setVisionSensorImg(sensor2Handle, image)

    # for x, y in zip(x_values, y_values):
    # sim.setGraphUserData(graph_handle, 'x', x)
    # sim.setGraphUserData(graph_handle, 'y', y)
    # sim.handleGraph(graph_handle, sim.getSimulationTime()) 
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    print('hi')
    sleep(1)
    sim.stopSimulation()
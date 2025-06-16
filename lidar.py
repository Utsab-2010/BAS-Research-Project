import sim

def sysCall_init():
    global sensor_handles
    sensor_handles = []
    for i in range(360):  # Assuming 360 sensors for a full rotation
        sensor_handle = sim.getObjectHandle(f"Proximity_sensor_{i}")
        sensor_handles.append(sensor_handle)

def sysCall_sensing():
    distances = []
    for handle in sensor_handles:
        detected, distance, _, _, _ = sim.readProximitySensor(handle)
        if detected:
            distances.append(distance)
        else:
            distances.append(float('nan'))  # No detection
    print(distances)

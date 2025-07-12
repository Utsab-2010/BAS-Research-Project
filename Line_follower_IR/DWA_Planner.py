import math
import numpy as np

class DWAPlanner:
    def __init__(self):
        # Robot dynamics limits
        self.v_max = 2.0
        self.v_min = 0.8
        self.omega_max = 1.5
        self.omega_min = -1.5
        self.a_max = 3.0
        self.alpha_max = 3.0

        # Simulation parameters
        self.dt = 0.1  # timestep
        self.predict_time = 0.8

        # Resolution
        self.v_res = 0.05
        self.omega_res = 0.1

        # Cost weights
        self.heading_weight = 1.0
        self.velocity_weight = 1.0
        self.obstacle_weight = 1.0

    def calc_dynamic_window(self, v_curr, omega_curr):
        return {
            'v_min': max(self.v_min, v_curr - self.a_max * self.dt),
            'v_max': min(self.v_max, v_curr + self.a_max * self.dt),
            'omega_min': max(self.omega_min, omega_curr - self.alpha_max * self.dt),
            'omega_max': min(self.omega_max, omega_curr + self.alpha_max * self.dt),
        }

    def simulate_trajectory(self, x_init, v, omega):
        traj = []
        x = list(x_init)  # [x, y, theta]
        for _ in np.arange(0, self.predict_time, self.dt):
            traj.append(tuple(x))
            x[0] += v * math.cos(x[2]) * self.dt
            x[1] += v * math.sin(x[2]) * self.dt
            x[2] += omega * self.dt
        return traj

    def heading_cost(self, traj, goal):
        last = traj[-1]
        dx = goal[0] - last[0]
        dy = goal[1] - last[1]
        angle_to_goal = math.atan2(dy, dx)
        heading_error = abs(math.atan2(math.sin(angle_to_goal - last[2]), math.cos(angle_to_goal - last[2])))
        return heading_error

    def velocity_cost(self, v):
        return self.v_max - v

    def obstacle_cost(self, traj, laser_points, clearance=0.5):
        min_dist = float('inf')
        for pose in traj:
            for pt in laser_points:
                dx = pt[0] - pose[0]
                dy = pt[1] - pose[1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
        if min_dist < clearance:
            return 1e6  # Collision imminent
        return 1.0 / min_dist

    def plan(self, current_pose, current_velocity, goal, laser_points):
        v_curr, omega_curr = current_velocity
        dw = self.calc_dynamic_window(v_curr, omega_curr)

        best_score = float('inf')
        best_v = 0.0
        best_omega = 0.0

        for v in np.arange(dw['v_min'], dw['v_max'] + self.v_res, self.v_res):
            for omega in np.arange(dw['omega_min'], dw['omega_max'] + self.omega_res, self.omega_res):
                traj = self.simulate_trajectory(current_pose, v, omega)

                h_cost = self.heading_cost(traj, goal)
                v_cost = self.velocity_cost(v)
                o_cost = self.obstacle_cost(traj, laser_points)

                total_cost = (self.heading_weight * h_cost +
                              self.velocity_weight * v_cost +
                              self.obstacle_weight * o_cost)

                if total_cost < best_score:
                    best_score = total_cost
                    best_v = v
                    best_omega = omega

        return best_v, best_omega

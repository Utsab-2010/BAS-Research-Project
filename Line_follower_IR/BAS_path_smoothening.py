import numpy as np


class BAS_Smoothner:
    def __init__(self):

        pass

    @staticmethod 
    def path_cost( ref_path, curr_path, costmap, alpha=0.5, beta=0.08, gamma=0.12):
        new_path = curr_path.copy()

        prev = new_path[:-2]
        curr = new_path[1:-1]
        next_ = new_path[2:]

        # Smoothness cost: encourages p_i ≈ (p_{i-1} + p_{i+1}) / 2
        smooth_term = prev - 2 * curr + next_
        smooth_cost = np.sum(np.linalg.norm(smooth_term, axis=1) ** 2)

        # Fidelity cost: stay close to reference (original A* path)
        fidelity_term = ref_path[1:-1] - curr
        fidelity_cost = np.sum(np.linalg.norm(fidelity_term, axis=1) ** 2)

        # Obstacle cost: higher near obstacles
        obstacle_cost = 0.0
        for p in curr:
            mx, my = int(p[0]), int(p[1])
            if 1 < mx < costmap.shape[1] - 2 and 1 < my < costmap.shape[0] - 2:
                cost = costmap[mx,my]
                obstacle_cost += cost

        # Weighted total cost
        total_cost = alpha * smooth_cost + beta * fidelity_cost + gamma * obstacle_cost
        print(f"Total cost: {total_cost}")
        return total_cost




    @staticmethod
    def smooth_path(path, costmap, alpha=0.1, beta=0.3, gamma=0.05,d = 3,D=1.5, iterations=100):
        ref_path = np.array(path)
        new_path = np.array(path)
        dim = new_path.shape[0]*new_path.shape[1]
        b = np.random.randn(dim)
        b = b / np.linalg.norm(b) # normalised random vector
        b = b.reshape(new_path.shape[0], new_path.shape[1])
        d_min = 0.01*d
        for _ in range(iterations):
            npath_l = new_path - b*d 
            npath_r = new_path + b*d

            new_path = new_path +  D*b*np.sign(BAS_Smoothner.path_cost(ref_path, npath_l, costmap, alpha, beta, gamma) 
                                 - BAS_Smoothner.path_cost(ref_path, npath_r, costmap, alpha, beta, gamma))
            
            d = 0.95*d + d_min
            D = 0.95*D 
        return new_path

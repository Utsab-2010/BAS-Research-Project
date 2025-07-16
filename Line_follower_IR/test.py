import heapq
import numpy as np

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    new_cost = cost + np.hypot(dx, dy)
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor, path + [neighbor]))
    return []

def subsample_path(path, step=3):
    if len(path) == 0:
        return []
    return path[::step] + [path[-1]]

# Example inputs
grid = np.zeros((10, 10), dtype=int)
grid[3:7, 5] = 1  # vertical wall

start = (0, 0)
goal = (9, 9)

path = astar(grid, start, goal)
waypoints = subsample_path(path, step=2)

print("Full path:", path)
print("Waypoints:", waypoints)

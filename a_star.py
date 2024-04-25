import math
import matplotlib.pyplot as plt

show_animation = True
pause_time = 0.001

class AStarPlanner:

    def __init__(self, obstacles_x, obstacles_y, resolution, robot_radius):
        """
        Initialize A* path planner.
        """
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.grid_width, self.grid_height = 0, 0
        self.motion = self.get_motion_model()
        self.create_obstacle_map(obstacles_x, obstacles_y)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # Grid index x
            self.y = y  # Grid index y
            self.cost = cost  # Cost to reach this node
            self.parent_index = parent_index  # Index of parent node

    def planning(self, start_x, start_y, goal_x, goal_y):
        """
        A* path search.
        """
        start_node = self.Node(self.calc_grid_index(start_x, self.min_x),
                               self.calc_grid_index(start_y, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_grid_index(goal_x, self.min_x),
                              self.calc_grid_index(goal_y, self.min_y), 0.0, -1)

        open_set, closed_set = {}, {}
        open_set[self.calc_node_key(start_node)] = start_node

        while open_set:
            current_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current_node = open_set[current_id]

            if show_animation:
                plt.plot(self.calc_grid_position(current_node.x, self.min_x),
                         self.calc_grid_position(current_node.y, self.min_y), "xc")
                plt.pause(0.001)

            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                print("Yeah Finally our goal reached.")
                goal_node.parent_index = current_node.parent_index
                goal_node.cost = current_node.cost
                break

            del open_set[current_id]
            closed_set[current_id] = current_node

            for move in self.motion:
                next_node = self.Node(current_node.x + move[0], current_node.y + move[1],
                                      current_node.cost + move[2], current_id)
                next_id = self.calc_node_key(next_node)

                if not self.is_node_valid(next_node):
                    continue

                if next_id in closed_set:
                    continue

                if next_id not in open_set or open_set[next_id].cost > next_node.cost:
                    open_set[next_id] = next_node

        rx, ry = self.trace_final_path(goal_node, closed_set)
        return rx, ry

    def trace_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_id = goal_node.parent_index
        while parent_id != -1:
            node = closed_set[parent_id]
            rx.append(self.calc_grid_position(node.x, self.min_x))
            ry.append(self.calc_grid_position(node.y, self.min_y))
            parent_id = node.parent_index
        return rx, ry

    @staticmethod
    def calc_heuristic(node1, node2):
        w = 1.0  # Heuristic weight
        return w * math.hypot(node1.x - node2.x, node1.y - node2.y)

    def calc_grid_position(self, index, min_position):
        """Calculate grid position in real-world coordinates.
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_grid_index(self, position, min_position):
        """
        Calculate grid index from real-world coordinate.
        """
        return round((position - min_position) / self.resolution)

    def calc_node_key(self, node):
        """
        Calculate a unique key for a node in the grid.
        """
        return (node.y - self.min_y) * self.grid_width + (node.x - self.min_x)

    def is_node_valid(self, node):
        """
        Check if a node is within the grid and not colliding with obstacles.
        """
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    def create_obstacle_map(self, obstacles_x, obstacles_y):
        """
        Create obstacle map from obstacle coordinates.
        """
        self.min_x = round(min(obstacles_x))
        self.min_y = round(min(obstacles_y))
        self.max_x = round(max(obstacles_x))
        self.max_y = round(max(obstacles_y))

        self.grid_width = round((self.max_x - self.min_x) / self.resolution)
        self.grid_height = round((self.max_y - self.min_y) / self.resolution)

        # Initialize obstacle map
        self.obstacle_map = [[False for _ in range(self.grid_height)] for _ in range(self.grid_width)]

        # Mark grid cells with obstacles
        for ix in range(self.grid_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.grid_height):
                y = self.calc_grid_position(iy, self.min_y)
                for ox, oy in zip(obstacles_x, obstacles_y):
                    if math.hypot(ox - x, oy - y) <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # Define motion model: [dx, dy, cost]
        return [[1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
                [-1, -1, math.sqrt(2)],
                [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)],
                [1, 1, math.sqrt(2)]]


def main():
    print(__file__ + " start!!")

    # Define start and goal positions
    start_x, start_y = 10.0, 10.0  # [m]
    goal_x, goal_y = 50.0, 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # Define obstacle positions
    obstacles_x, obstacles_y = [], []
    for i in range(-10, 60):
        obstacles_x.append(i)
        obstacles_y.append(-10.0)
    for i in range(-10, 60):
        obstacles_x.append(60.0)
        obstacles_y.append(i)
    for i in range(-10, 61):
        obstacles_x.append(i)
        obstacles_y.append(60.0)
    for i in range(-10, 61):
        obstacles_x.append(-10.0)
        obstacles_y.append(i)
    for i in range(-10, 40):
        obstacles_x.append(20.0)
        obstacles_y.append(i)
    for i in range(0, 40):
        obstacles_x.append(40.0)
        obstacles_y.append(60.0 - i)

    if show_animation:
        plt.plot(obstacles_x, obstacles_y, ".k")
        plt.plot(start_x, start_y, "og")
        plt.plot(goal_x, goal_y, "xb")
        plt.grid(True)
        plt.axis("equal")
        label_column = ['Start', 'Goal', 'Path taken',
                        'Current computed path',
                        'Obstacles']
        columns = [plt.plot([], [], symbol, color=colour, alpha=alpha)[0]
                   for symbol, colour, alpha in [['o', 'g', 1],
                                                 ['x', 'b', 1],
                                                 ['-', 'r', 1],
                                                 ['.', 'c', 1],
                                                 ['.', 'k', 1]]]
        plt.legend(columns, label_column, bbox_to_anchor=(1, 1), title="Key:",
                   fontsize="xx-small")
        plt.plot()
        plt.pause(pause_time)

    # Initialize A* planner
    planner = AStarPlanner(obstacles_x, obstacles_y, grid_size, robot_radius)
    rx, ry = planner.planning(start_x, start_y, goal_x, goal_y)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()

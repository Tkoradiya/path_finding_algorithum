import numpy as np
import matplotlib.pyplot as plt

SHOW_ANIMATION = True
USE_BEAM_SEARCH = False
USE_ITERATIVE_DEEPENING = False
USE_DYNAMIC_WEIGHTING = False
USE_THETA_STAR = False
USE_JUMP_POINT = False

BEAM_CAPACITY = 30
MAX_THETA = 5
ONLY_CORNERS = False
MAX_CORNER = 5
W, EPSILON, UPPER_BOUND_DEPTH = 1, 4, 500
pause_time = 0.001


def draw_line(start_x, start_y, length, orientation, o_x, o_y, o_dict):
    """
    Draw a horizontal or vertical line of obstacles on the grid.

    Args:
        start_x (int): Starting x-coordinate.
        start_y (int): Starting y-coordinate.
        length (int): Length of the line.
        orientation (str): Orientation of the line ('horizontal' or 'vertical').
        o_x (list): List to store x-coordinates of obstacles.
        o_y (list): List to store y-coordinates of obstacles.
        o_dict (dict): Dictionary to mark obstacle positions.

    """
    if orientation == 'horizontal':
        for i in range(start_x, start_x + length):
            for j in range(start_y, start_y + 2):
                o_x.append(i)
                o_y.append(j)
                o_dict[(i, j)] = True
    elif orientation == 'vertical':
        for i in range(start_x, start_x + 2):
            for j in range(start_y, start_y + length):
                o_x.append(i)
                o_y.append(j)
                o_dict[(i, j)] = True


def is_line_of_sight(obs_grid, x1, y1, x2, y2):
    """
    Check if there's a clear line of sight between two points.

    Args:
        obs_grid (dict): Dictionary representing obstacle grid.
        x1 (int): x-coordinate of start point.
        y1 (int): y-coordinate of start point.
        x2 (int): x-coordinate of end point.
        y2 (int): y-coordinate of end point.

    Returns:
        tuple: Tuple containing a boolean indicating line of sight and the distance if clear.

    """
    t = 0
    while t <= 0.5:
        xt = (1 - t) * x1 + t * x2
        yt = (1 - t) * y1 + t * y2
        if obs_grid.get((int(xt), int(yt)), False):
            return False, None
        xt = (1 - t) * x2 + t * x1
        yt = (1 - t) * y2 + t * y1
        if obs_grid.get((int(xt), int(yt)), False):
            return False, None
        t += 0.001
    dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    return True, dist


def compute_key_points(o_dict):
    """
    Compute key points (corners) on the obstacle grid.

    Args:
        o_dict (dict): Dictionary representing obstacle grid.

    Returns:
        list: List of key points (corners) on the grid.

    """
    offsets1 = [(1, 0), (0, 1), (-1, 0), (1, 0)]
    offsets2 = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    offsets3 = [(0, 1), (-1, 0), (0, -1), (0, -1)]
    corner_list = []

    for grid_point, obs_status in o_dict.items():
        if obs_status:
            continue
        empty_space = True
        x, y = grid_point

        # Check if the point is surrounded by obstacles or boundary
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (x + i, y + j) not in o_dict:
                    continue
                if o_dict[(x + i, y + j)]:
                    empty_space = False
                    break
            if not empty_space:
                break

        if empty_space:
            continue

        for offset1, offset2, offset3 in zip(offsets1, offsets2, offsets3):
            i1, j1 = offset1
            i2, j2 = offset2
            i3, j3 = offset3

            if ((x + i1, y + j1) not in o_dict) or \
               ((x + i2, y + j2) not in o_dict) or \
               ((x + i3, y + j3) not in o_dict):
                continue

            obs_count = sum(1 for offset in [offset1, offset2, offset3] if o_dict[(x + offset[0], y + offset[1])])

            if obs_count == 3 or obs_count == 1:
                corner_list.append((x, y))
                if SHOW_ANIMATION:
                    plt.plot(x, y, ".y")

    if ONLY_CORNERS:
        return corner_list

    edge_list = []
    for corner in corner_list:
        x1, y1 = corner
        for other_corner in corner_list:
            x2, y2 = other_corner
            if x1 == x2 and y1 == y2:
                continue
            reachable, _ = is_line_of_sight(o_dict, x1, y1, x2, y2)
            if reachable:
                x_m, y_m = int((x1 + x2) / 2), int((y1 + y2) / 2)
                edge_list.append((x_m, y_m))
                if SHOW_ANIMATION:
                    plt.plot(x_m, y_m, ".y")
    return corner_list + edge_list


class SearchAlgorithm:
    def __init__(self, obs_grid, goal_x, goal_y, start_x, start_y, limit_x, limit_y, corner_list=None):
        self.start_pt = [start_x, start_y]
        self.goal_pt = [goal_x, goal_y]
        self.obs_grid = obs_grid
        self.all_nodes, self.open_set = {}, []

        if USE_JUMP_POINT:
            for corner in corner_list:
                i, j = corner
                h_c = self.get_heuristic(i, j, goal_x, goal_y)
                self.all_nodes[(i, j)] = {'pos': [i, j], 'pred': None,
                                          'gcost': np.inf, 'hcost': h_c,
                                          'fcost': np.inf,
                                          'open': True, 'in_open_list': False}
            self.all_nodes[tuple(self.goal_pt)] = {'pos': self.goal_pt, 'pred': None,
                                                    'gcost': np.inf, 'hcost': 0, 'fcost': np.inf,
                                                    'open': True, 'in_open_list': True}
        else:
            for i in range(limit_x):
                for j in range(limit_y):
                    h_c = self.get_heuristic(i, j, goal_x, goal_y)
                    self.all_nodes[(i, j)] = {'pos': [i, j], 'pred': None,
                                              'gcost': np.inf, 'hcost': h_c,
                                              'fcost': np.inf,
                                              'open': True, 'in_open_list': False}
        self.all_nodes[tuple(self.start_pt)] = {'pos': self.start_pt, 'pred': None,
                                                'gcost': 0, 'hcost': self.get_heuristic(start_x, start_y, goal_x, goal_y),
                                                'fcost': 0 + self.get_heuristic(start_x, start_y, goal_x, goal_y),
                                                'open': True, 'in_open_list': True}
        self.open_set.append(self.all_nodes[tuple(self.start_pt)])

    def get_heuristic(self, x1, y1, x2, y2):
        """
        Calculate the heuristic (Manhattan distance) from (x1, y1) to (x2, y2).

        Args:
            x1 (int): x-coordinate of start point.
            y1 (int): y-coordinate of start point.
            x2 (int): x-coordinate of goal point.
            y2 (int): y-coordinate of goal point.

        Returns:
            int: Heuristic value (Manhattan distance).

        """
        return abs(x1 - x2) + abs(y1 - y2)

    def a_star(self):
        """
        Perform A* search algorithm.

        """
        if SHOW_ANIMATION:
            plt.title('A*')

        goal_found = False
        while self.open_set:
            self.open_set = sorted(self.open_set, key=lambda x: (x['fcost'], x['hcost'], x['gcost']))
            current_node = self.all_nodes[tuple(self.open_set.pop(0)['pos'])]

            if tuple(current_node['pos']) == tuple(self.goal_pt):
                goal_found = True
                break

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (i == 0 and j == 0) or \
                       (current_node['pos'][0] + i, current_node['pos'][1] + j) not in self.obs_grid:
                        continue

                    neighbor_pos = (current_node['pos'][0] + i, current_node['pos'][1] + j)
                    neighbor_node = self.all_nodes.get(neighbor_pos)

                    if not neighbor_node or self.obs_grid[neighbor_pos]:
                        continue

                    g_cost = current_node['gcost'] + (10 if i == 0 or j == 0 else 14)
                    h_cost = neighbor_node['hcost']
                    f_cost = g_cost + h_cost

                    if f_cost < neighbor_node['fcost']:
                        neighbor_node['pred'] = current_node['pos']
                        neighbor_node['gcost'] = g_cost
                        neighbor_node['fcost'] = f_cost

                        if not neighbor_node['in_open_list']:
                            self.open_set.append(neighbor_node)
                            neighbor_node['in_open_list'] = True

            current_node['open'] = False

        if SHOW_ANIMATION:
            while goal_found:
                if current_node['pred'] is None:
                    break
                plt.plot([current_node['pos'][0], current_node['pred'][0]],
                         [current_node['pos'][1], current_node['pred'][1]], "b")
                current_node = self.all_nodes[tuple(current_node['pred'])]
                plt.pause(0.001)
            plt.show()


def main():
    # Set obstacle positions
    obs_dict = {(i, j): False for i in range(51) for j in range(51)}

    s_x, s_y = 5.0, 5.0
    g_x, g_y = 35.0, 45.0

    # Draw outer border of maze
    draw_line(0, 0, 50, 'vertical', [], [], obs_dict)
    draw_line(48, 0, 50, 'vertical', [], [], obs_dict)
    draw_line(0, 0, 50, 'horizontal', [], [], obs_dict)
    draw_line(0, 48, 50, 'horizontal', [], [], obs_dict)

    
    # Draw inner walls
    walls = [(10, 10, 10), (10, 30, 10), (10, 45, 5), (15, 20, 10), (20, 5, 10), (20, 40, 5),
             (30, 10, 20), (30, 40, 10), (35, 5, 25), (30, 40, 10), (40, 10, 35), (45, 25, 15)]
    for x, y, length in walls:
        draw_line(x, y, length, 'vertical', [], [], obs_dict)

    walls = [(35, 5, 10), (40, 10, 5), (15, 15, 10), (10, 20, 10), (45, 20, 5), (20, 25, 5),
             (10, 30, 10), (15, 35, 5), (25, 35, 10), (45, 35, 5), (10, 40, 10), (30, 40, 5),
             (10, 45, 5), (40, 45, 5)]
    for x, y, length in walls:
        draw_line(x, y, length, 'horizontal', [], [], obs_dict)

    # Plotting obstacles
    o_x = [coord[0] for coord, is_obs in obs_dict.items() if is_obs]
    o_y = [coord[1] for coord, is_obs in obs_dict.items() if is_obs]

    if SHOW_ANIMATION:
        plt.plot(o_x, o_y, ".k")
        plt.plot(s_x, s_y, "og")
        plt.plot(g_x, g_y, "xb")
        plt.grid(True)
        plt.axis("equal")
        label_column = ['Start', 'Goal', 'Path taken',
                        'Obstacles']
        columns = [plt.plot([], [], symbol, color=colour, alpha=alpha)[0]
                   for symbol, colour, alpha in [['o', 'g', 1],
                                                 ['x', 'b', 1],
                                                 ['-', 'b', 1],
                                                 ['.', 'k', 1]]]
        plt.legend(columns, label_column, bbox_to_anchor=(1, 1), title="Key:",
                   fontsize="xx-small")
        plt.plot()
        plt.pause(pause_time)

    if USE_JUMP_POINT:
        corner_list = compute_key_points(obs_dict)
        search_algo = SearchAlgorithm(obs_dict, g_x, g_y, s_x, s_y, 101, 101, corner_list)
        search_algo.a_star()
    else:
        search_algo = SearchAlgorithm(obs_dict, g_x, g_y, s_x, s_y, 101, 101)
        search_algo.a_star()

if __name__ == '__main__':
    main()


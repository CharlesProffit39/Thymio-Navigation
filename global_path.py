"""
File name: global_path.py
Authors: Jin Kilidjian, Corentin Plumet, Jules Bervillé, Charles Proffit 
Created: 05/12/24
Description: path planning functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import math
import heapq
import parameters as params

# #########################################
#             #A-STAR
# #########################################
"""
https://www.geeksforgeeks.org/a-search-algorithm-in-python/
"""
# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination



# Check if a cell is valid (within the grid)
def is_valid(row, col):
    return (row >= 0) and (row < params.MAP_WIDTH) and (col >= 0) and (col < params.MAP_LENGTH)

# Check if a cell is unblocked
def is_unblocked(grid, row, col):
    return grid[row][col] == 0

# Check if a cell is the destination
def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# Calculate the heuristic value of a cell (Euclidean distance to destination)
def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

# Trace the path from source to destination
def trace_path(cell_details, dest):
    print("The Path is ")
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()

    # Print the path
    for i in path:
        print("->", i, end=" ")
    print()
    return path

# Implement the A* search algorithm
def a_star_search(grid, src, dest):
    src = [int(src[0]/params.SIZE_CASE),int(src[1]/params.SIZE_CASE)]
    dest = [int(dest[0]/params.SIZE_CASE),int(dest[1]/params.SIZE_CASE)]

    # Check if the source and destination are valid
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        return

    # Check if the source and destination are unblocked
    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        print("Source or the destination is blocked")
        return

    # Check if we are already at the destination
    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return

    # Initialize the closed list (visited cells)
    closed_list = [[False for _ in range(params.MAP_LENGTH)] for _ in range(params.MAP_WIDTH)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(params.MAP_LENGTH)] for _ in range(params.MAP_WIDTH)]

    # Initialize the start cell details
    i = src[0]
    j = src[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    # Initialize the open list (cells to be visited) with the start cell
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    # Initialize the flag for whether destination is found
    found_dest = False

    # Main loop of A* search algorithm
    while len(open_list) > 0:
        # Pop the cell with the smallest f value from the open list
        p = heapq.heappop(open_list)

        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        # For each direction, check the successors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            # If the successor is valid, unblocked, and not visited
            if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                # If the successor is the destination
                if is_destination(new_i, new_j, dest):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    print("The destination cell is found")
                    # Trace and print the path from source to destination
                    path = trace_path(cell_details, dest)
                    found_dest = True
                    # path = gp.sampling(path,sampling=params.SAMPLING)
                    # gp.plot_map(padded_map, path)
                    # path = gp.add_angle_to_path(path)
                    # path = gp.convert_to_mm(path) #path in mm x[mm],y[mm]
                    return path
                else:
                    # Calculate the new f, g, and h values
                    if dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        g_new = cell_details[i][j].g + np.sqrt(2)
                    else:
                        g_new = cell_details[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the cell is not in the open list or the new f value is smaller
                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the cell details
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

    # If the destination is not found after visiting all cells
    if not found_dest:
        print("Failed to find the destination cell")

#########################################
            #HELPERS
#########################################


def plot_map(grid, path=None, simu_robot_trajectory=None):
    """
    Description
    -----------
    Plot the map, the global path, the simulated trajectory and initialize the real trajectory

    Parameters
    ----------
    grid (list of list): the grid captured by the camera
    path (list of tuple): the path computed by the a-star
    simu_robot_trajectory (list of tuple): the path followed by the robot in simulation

    Return
    ------
    fig (plt.subplot): figure to plot
    line (ax.plot): the line object containing the robot real trajectory
    ellipse (matplotlib.patches.Ellipse): the ellipse object containing the visual variance for the graph   
    """
    # Define the colors used on the map
    # White for 0, Gray for 0.5, Black for 1
    color_map = ListedColormap(["white", "gray", "black"])

    #Variance ellipse
    ellipse = Ellipse(
        xy=(0,0),       
        width=0,         
        height=0,       
        angle=0,         
        edgecolor='blue',  
        facecolor='none',  
        linewidth=2
    )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap=color_map, vmin=0, vmax=1, origin="upper")
    ax.add_patch(ellipse)

    # A* path in RED
    if path is not None:
        path_x, path_y, angle = zip(*path)
        ax.plot(path_y, path_x, '-r', label="A* Path", linewidth=2)
    
    # robot's path in BLUE
    if simu_robot_trajectory is not None:
        traj_x, traj_y = zip(*simu_robot_trajectory) 
        ax.plot(traj_y, traj_x, '-b', label="Simulated trajectory", linewidth=2)   

    #Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([]) 

    ax.set_xlabel("Y")
    ax.set_ylabel("X") 
    ax.set_title("Grid Map with Path and Robot Trajectory")

    line, = ax.plot([],[], color = 'lightgreen', label="Real trajectory", linewidth=2)
    ax.legend() 

    return fig,line,ellipse

def padding(grid):
    """
    Description
    -----------
    Make the obstacle bigger and draw a forbidden region on the edge of the map

    Parameters
    ----------
    grid (list of list): the grid captured by the camera

    Globals
    -------
    int PADDING_RATI0 : Padding ratio compared to the robot size
    int PADDING_SIDE : size of the padding for the edge of the map
    int SIZE_CASE : length of the side of a case
    int AXLE_LENGTH : length of the wheels axle
    
    Return
    ------
    grid (list of list): the modified grid  
    """
    depth_obstacle = int(params.PADDING_RATI0*(params.AXLE_LENGTH/2)/params.SIZE_CASE)
    for i, row in enumerate(grid):
        for j, element in enumerate(row):
            if element == 1:
                for x in range (-depth_obstacle, depth_obstacle +1):
                    for y in range (-depth_obstacle, depth_obstacle +1):
                        if i+x <= len(grid)-1 and j+y <= len(grid[0]) -1:
                            if grid[i+x][j+y] != 1:
                                grid[i+x][j+y] = 0.5

            elif element == 0:
                if i < params.PADDING_SIDE or i + params.PADDING_SIDE >= len(grid) or j < params.PADDING_SIDE or j + params.PADDING_SIDE >= len(grid[0]):
                    grid[i][j] = 0.5

    return grid

def add_angle_to_path(path):
    """
    Description
    -----------
    Add a third value to the each point of the path -> the angle to follow to attain the next point of the path 

    Parameters
    ----------
    path (list of tuple): the path computed by the a-star

    Return
    ------
    new_path (list of tuple): the path computed by the a-star with a third value in the tuple (angle)  
    """
    new_path = []

    for i in range(len(path)):
        if i !=0:
            angle = math.atan2(path[i][1] - path[i-1][1], path[i][0] - path[i-1][0])
        else:
            angle = math.atan2(path[i+1][1] - path[i][1], path[i+1][0] - path[i][0])  #first angle different

        new_path.append((path[i][0], path[i][1], angle))

    return new_path


def sampling(path, n=2):
    """
    Description
    -----------
    Sample the path points

    Parameters
    ----------
    path (list of tuple): the path computed by the a-star
    n (int): the sampling frequency (1 point sampled out of n point)

    Return
    ------
    sampled_path (list of tuple): the sampled path  
    """
    sampled_path = []

    for i, element in enumerate(path):
        if i % n == 0:
            sampled_path.append(element)
        elif i == len(path) - 1:
            sampled_path.append(element)

    return sampled_path

def convert_to_mm(path):
    """
    Description
    -----------
    Convert to mm the path to go from grid coordinata to real world coordinate

    Parameters
    ----------
    path (list of tuple): the path computed by the a-star
    
    Globals
    -------
    int SIZE_CASE : length of the side of a case

    Return
    ------
    (list of list): the same path but with each point being in mm 
    """
    return [[sublist[0] * params.SIZE_CASE, sublist[1] * params.SIZE_CASE, *sublist[2:]] for sublist in path]

#### IF ANGLE IN THE LIST ####
def extract_direction_change(path):
    """
    Description
    -----------
    Extract from the path the point where the robot changes direction

    Parameters
    ----------
    path (list of tuple): the path computed by the a-star

    Return
    ------
    new_path (list of tuple): the list of point where the robot changes direction
    """
    new_path = []
    previous_angle = None

    for i, element in enumerate(path):
        if element[2] != previous_angle:
            new_path.append(element)

        previous_angle = element[2]

    return new_path

def euclidean_distance(x1, y1, x2, y2):
    """
    Description
    -----------
    Compute the euclidean distance between 2 points

    Parameters
    ----------
    x1 (int): x1 coordinate
    y1 (int): y1 coordinate
    x2 (int): x2 coordinate
    y2 (int): y2 coordinate

    Return
    ------
    new_path (list of tuple): the list of point where the robot changes direction
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class Simu_Robot:
    """
    Description
    -----------
    Class to simulate the robot behavior
    """
    def __init__(self, state = [0, 0, 0], v_left=0.0, v_right=0.0):
        """
        Description
        -----------
        Initialize the Simu_Robot object
        """
        self.state = state
        self.v_left = v_left  # Left wheel velocity
        self.v_right = v_right  # Right wheel velocity

    def control_to_target(self,goal,last_segment):
        """
        Description
        -----------
        Steers the motors in order to reach the goal/next node with desired angle
        
        Parameters
        ----------
        list of float state : state[0],state[1] (x and y positions of the robot), state[2] (angle of the robot w.r.t the x-axis)
        list of float goal : goal[0],goal[1] (x and y positions of the goal/next node), goal[2] (desired angle of the robot when reaching the goal/next node)
        bool last_segment : activates or not the slow down for the last segment to the goal
        
        Globals
        -------
        float K_DIST, K_ALPHA, K_BETA : parameters of the Astolfi control law
        int NOMINAL_SPEED : linear speed of the center of rotation of the robot
        int AXLE_LENGTH : length of the wheels axle
        
        Return
        ------
        None
        """
        x_dir = (goal[0] - self.state[0])

        y_dir = (goal[1] - self.state[1])
        theta_dir = np.arctan2(y_dir,x_dir)
        # Using wrapping angle, we can keep alpha and beta between [-pi;pi]
        alpha = (theta_dir - self.state[2] + np.pi) % (2*np.pi) - np.pi
        beta =  (goal[2] - self.state[2] - alpha + np.pi) % (2*np.pi) - np.pi
        # alpha = theta_dir - theta
        # beta =  theta_goal - theta_dir
        dist = np.sqrt(x_dir**2 + y_dir**2)
        if last_segment:
            v = params.K_DIST*dist
            if v < params.MIN_SPEED:
                v = params.MIN_SPEED
        else:
            v = params.NOMINAL_SPD
        w = params.K_ALPHA*alpha + params.K_BETA*beta
        # if alpha > np.pi / 2 or alpha < -np.pi / 2:
        #     v = -v
        self.v_left = (v - params.AXLE_LENGTH*w/2)*params.REDUC_COEF/params.SIZE_CASE
        self.v_right = (v + params.AXLE_LENGTH*w/2)*params.REDUC_COEF/params.SIZE_CASE


    def update(self):
        """
        Description
        -----------
        Update the new position of the robot

        Parameters
        ----------
        v_right (int): the velocity of the right wheel
        v_left (int): the velocity of the left wheel
        state (list): the position and angle of the robot
        
        Globals
        -------
        int SIZE_CASE : length of the side of a case
        int AXLE_LENGTH : length of the wheels axle
        int DT : time step of the simulation
        
        Return
        ------
        None
        """
        # Angular velocity of the robot
        velocity_angle = (self.v_right - self.v_left) / ((params.AXLE_LENGTH)/params.SIZE_CASE)  
         # Linear velocity of the robot
        linear_velocity = (self.v_left + self.v_right) / 2 

        # Update x, y and yaw (update state)
        self.state[0] = self.state[0] + linear_velocity * math.cos(self.state[2]) * params.DT
        self.state[1] = self.state[1] + linear_velocity * math.sin(self.state[2]) * params.DT
        self.state[2] = self.state[2] + velocity_angle * params.DT

def simu(path, padded_grid, start=[0,0,0]):
    """
    Description
    -----------
    Simulate the trajectory of the robot

    Parameters
    ----------
    path (list of tuple): the path computed by the a-star
    padded_grid (list of list): the grid captured by the camera with the padding
    start (list): the initial position of the robot (x,y and angle)
    
    Globals
    -------
    int SIZE_CASE : length of the side of a case
    int DIST_ERROR : the admissible error between the robot and the objective point
    
    Return
    ------
    """
    start = [i/params.SIZE_CASE for i in start]

    start[2] = (start[2]+np.pi/2)%(2*np.pi)

    simu_path = []
    robot = Simu_Robot(state=[start[0], start[1], start[2]])

    for target in path:
        last_segment = False
        while euclidean_distance(robot.state[0], robot.state[1], target[0], target[1]) > params.DIST_ERROR/params.SIZE_CASE:  
            if target == path[-1]:
                last_segment = True

            robot.control_to_target(target, last_segment)
            robot.update()
            simu_path.append((robot.state[0], robot.state[1]))
    
    fig, line,ellipse = plot_map(padded_grid, path, simu_path)

    return fig, line, ellipse
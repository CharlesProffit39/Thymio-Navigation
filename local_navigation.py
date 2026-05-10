"""
File name: local_navigation.py
Authors: Jin Kilidjian, Corentin Plumet, Jules Bervillé, Charles Proffit 
Created: 05/12/24
Description: obstacle avoidance and kidnapping functions
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
import parameters as params

def object_detected(proxi):
    """
    Description
    -----------
    Detects if an obstacle is in front of the proximity sensors

    Parameters
    ----------
    list of int proxi : values of proximity sensors 1 to 8

    Globals
    -------
    int OBSTACLE_TRHESHOLD : intensity threshold for object detection

    Return
    ------
    bool object_detected(proxi): True if object is detected, False else    
    """
    for i in range(5):
        if proxi[i] > params.OBSTACLE_TRHESHOLD:
            return True
    return False

def local_avoidance(proxi):
    """
    Description
    -----------
    Steers the motors in order to avoid the obstacle detected by proximity sensors

    Parameters
    ----------
    list of int proxi : values of proximity sensors 1 to 7

    Globals
    -------
    int NOMINAL_SPD : mean speed of the robot
    list of int ANN_LEFT,ANN_RIGHT : weights for the ANN network of proximity sensors

    Return
    ------
    None
    """
    speed_left = params.NOMINAL_SPD
    speed_right = params.NOMINAL_SPD
    for i in range(7):        
        speed_left = speed_left + proxi[i] * params.ANN_LEFT[i]
        speed_right = speed_right + proxi[i] * params.ANN_RIGHT[i]
    return speed_left,speed_right

def get_odometry(spd_right,spd_left):
    """
    Description
    -----------
    Calculates the next linear and angular speeds of the robot given its motors speeds.

    Parameters
    ----------
    spd_right : linear speed of the right wheel [mm/s]
    spd_left : linear speed of the left wheel [mm/s]

    Globals
    -------
    int AXLE_LENGTH : length of the wheels axle [mm]

    Returns
    -------
    float v : next linear speed of the thymio [mm/s]
    float w : next angular speed of the thymio [rad/s] 
    """
    v = (spd_left+spd_right)*params.REDUC_COEF/2 #linear velocity of COR [mm/sec]
    w = (spd_right-spd_left)*params.REDUC_COEF/params.AXLE_LENGTH #angular velocity of COR [rad/sec]
    return v,w


def control(state,goal,last_segment):
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
    int MIN_SPEED : miminal speed of the Thymio
    int AXLE_LENGTH : length of the wheels axle
    
    Return
    ------
    None
    """

    x_dir = (goal[0] - state[0])
    y_dir = (goal[1] - state[1])
    theta_dir = np.arctan2(y_dir,x_dir)
    # Using wrapping angle, we can keep alpha and beta between [-pi;pi]
    alpha = (theta_dir - state[2] + np.pi) % (2*np.pi) - np.pi #error in angle between robot direction and direction to go to goal
    beta =  (goal[2] - state[2] - alpha + np.pi) % (2*np.pi) - np.pi #error in angle between robot direction to reach goal, and optimal direction to reach goal
    dist = np.sqrt(x_dir**2 + y_dir**2)
    if last_segment:
        v = params.K_DIST*dist
        if v < params.MIN_SPEED:
            v = params.MIN_SPEED
    else:
        v = params.NOMINAL_SPD
    w = params.K_ALPHA*alpha + params.K_BETA*beta
    v_left = v - params.AXLE_LENGTH*w/2
    v_right = v + params.AXLE_LENGTH*w/2
    return v_left,v_right

def ellipse_params(matrix):
    """
    Description
    -----------
    Calculate the ellipse parameters (major axis, minor axis, and rotation angle)

    Parameters
    ---------
    numpy array matrix: 2x2 covariance matrix [[a, b], [b, c]] of x and y estimates

    Return
    ------
    numpy float np.sqrt(lambda1), np.sqrt(lambda2) : major and minor axis 1-std length
    angle : angle of the ellipse
    """

    a = matrix[0, 0]
    b = matrix[0, 1] 
    c = matrix[1, 1]

    # Eigenvalues
    lambda1 = ((a+c) / 2) + np.sqrt(((a - c) / 2)**2 + b**2)  # Larger eigenvalue
    lambda2 = ((a+c) / 2) - np.sqrt(((a - c) / 2)**2 + b**2)  # Smaller eigenvalue

    # Compute rotation angle θ
    if b == 0:
        if a >= c:
            angle = 0
        else:
            angle = 180
    else:
        angle = np.arctan2(lambda1-a,b)*180/np.pi

    return np.sqrt(lambda1), np.sqrt(lambda2), angle


def update_plots(states,P,fig,line,ellipse):
    """
    Description
    -----------
    Update plots in live at each iteration
    
    Globals
    -------
    int SIZE_CASE : real length/width of a case in the plot [mm]

    Parameters
    ----------
    list of float states : states[0],states[1] (list of positions of the robot in x and y), states[2] (list of angles)
    fig : figure to update
    line : list of datas to plot
    ellipse : Ellipse object for the 2D ellipse plot of the variance in x and y 
    
    Return
    ------
    None
    """

    # Updating the variance ellipse
    x = states[-1][0]/params.SIZE_CASE
    y = states[-1][1]/params.SIZE_CASE
    ellipse.set_center((y,x))

    std_maj,std_min,ang = ellipse_params(P[:2, :2])
    ellipse.set_width(std_maj**(0.5)*3/params.SIZE_CASE) #3-std dev along major axis 
    ellipse.set_height(std_min**(0.5)*3/params.SIZE_CASE) #3-std dev along minor axis
    # ellipse.set_angle(ang*180/np.pi) #optional angle display

    # Updating the x-y trajectory
    xline = [sublist[0] for sublist in states]
    xline = [i/params.SIZE_CASE for i in xline]
    yline = [sublist[1] for sublist in states]
    yline = [i/params.SIZE_CASE for i in yline]
    line.set_data(yline,xline)
    fig.canvas.draw()
    fig.canvas.flush_events()  

def kidnapping_detected(prox_ground):
    """
    Description
    -----------
    Detects kidnapping scenarios (Thymio not on the ground, prox_ground giving very low values)

    Globals
    -------
    PROX_THRESHOLD : minimal value of prox.ground.reflected to enter kidnapping mode

    Parameters
    ----------
    list of int prox_ground : proximity sensors at the bottom (in Thymio units)
    
    Return
    ------
    boolean : True if kidnapped, false else
    """
    if prox_ground[0] < params.PROX_THRESHOLD and prox_ground[1] < params.PROX_THRESHOLD:
        return True
    return False

def goal_is_reached(states,nodes):
    """
    Description
    -----------
    Checks if overall goal is reached

    Parameters
    ----------
    list of (list of float) states : state[0],state[1] (x and y positions of the robot), state[2] (angle of the robot w.r.t the x-axis)
    list of (list of float) nodes : nodes[0],nodes[1] (x and y positions of the goal/next node), nodes[2] (desired angle of the robot when reaching the goal/next node)    
    
    Globals
    -------
    DIST_ERROR : Margin of error when reaching the node

    Return
    ------
    boolean : True if overall goal is reached, false else
    """
    return (np.sqrt((states[-1][0]-nodes[-1][0])**2 + (states[-1][1]-nodes[-1][1])**2) <= params.DIST_ERROR)

def node_is_reached(states,nodes,counter):
    """
    Description
    -----------
    Checks if next node is reached

    Parameters
    ----------
    list of (list of float) states : state[0],state[1] (x and y positions of the robot), state[2] (angle of the robot w.r.t the x-axis)
    list of (list of float) nodes : nodes[0],nodes[1] (x and y positions of the goal/next node), nodes[2] (desired angle of the robot when reaching the goal/next node)    
    
    Globals
    -------
    DIST_ERROR : Margin of error when reaching the node

    Return
    ------
    boolean : True if next node is reached, false else
    """
    return (np.sqrt((states[-1][0]-nodes[counter][0])**2 + (states[-1][1]-nodes[counter][1])**2) <= params.DIST_ERROR)


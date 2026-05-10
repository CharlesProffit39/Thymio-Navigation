"""
File name: parameters.py
Authors: Jin Kilidjian, Corentin Plumet, Jules Bervillé, Charles Proffit 
Created: 05/12/24
Description: magic numbers of our program
"""

## ---------GLOBAL DATAS (NOT TO BE MODIFIED, CONSTANTS)--------- ##

DT = 0.2   #[sec]
T_UPDATE_CAM = 1 #[sec]

## THYMIO PARAMETERS ##
REDUC_COEF = 0.3389 # real speed [mm/s] = thymio speed*reduc_coef
RADIUS = 20 #[mm] wheel radius
AXLE_LENGTH = 95 #[mm] axle length

## STATES OF THE FSM ##
FOLLOW_PATH = 0
OBSTACLE_AVOIDANCE = 1
KIDNAPPED = 2

## STATES OF THE CAMERA ##
NOMINAL = 0
OBSTRUCTED = 1


## ---------GLOBAL DATAS (TO BE MODIFIED)--------- ##

# DEFINE MAP AND VISION PARAMETERS
MAP_LENGTH = 297
MAP_WIDTH = 210
SIZE_CASE = 4 #[mm] size of each case of the grid
MAX_NB_OBSTACLES = 4
THRESHOLD = 100
PADDING_RATI0 = 1.5 #padding 1.6 time the half of the axle length
PADDING_SIDE = 5
SAMPLING = 20

## KALMAN PARAMETERS ##
Q_X = 1.7 #[mm²]
Q_Y = 1.7 #[mm²]
Q_THETA = 0.096 #[rad²]
Q_V = 0.43 #[mm²/s²]
Q_W = 0.0125 #[rad²/s²]
R_V =  0.283 #[mm²/s²]
R_W = 0.00125 #[rad²/s²]
R_X = 0.68 #[mm²]
R_Y = 0.68 #[mm²]
R_THETA = 0.019 #[rad²]

## LOCAL AVOIDANCE PARAMETERS ##
NOMINAL_SPD = 100       # nominal speed

## OBSTACLE AVOIDANCE PARAMETERS ##
OBSTACLE_TRHESHOLD = 20      # high obstacle threshold to switch state 0->1

ANN_LEFT = [0.04,  0.02, -0.02, -0.02, -0.04,  0.03, -0.01]
ANN_RIGHT = [-0.04, -0.02, -0.02,  0.02,  0.04, -0.01,  0.03]
ANN_LEFT = [i / 5 for i in ANN_LEFT]
ANN_RIGHT = [i / 5 for i in ANN_RIGHT]

PROX_THRESHOLD = 15  # height threshold before detecting a kidnapping scenario (robot on the back)

## ASTOLFI CONTROL PARAMETERS ##
K_DIST = 0.1 #coef for control to goal
K_ALPHA = 2.2 #coef for control error alpha
K_BETA = -0.16 #coef for control error beta
DIST_ERROR = 20 #[mm] maximal accepted error when reaching node
MIN_SPEED = 60

def set_threshold(val):
    global THRESHOLD
    THRESHOLD = val
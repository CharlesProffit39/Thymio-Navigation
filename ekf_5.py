"""
File name: ekf_5.py
Authors: Jin Kilidjian, Corentin Plumet, Jules Bervillé, Charles Proffit 
Created: 05/12/24
Description: Extended Kalman filter functions
"""

import numpy as np
import parameters as params

# P : state covariance matrix
# Q : state model covariance matrix
# R : sensor measurement covariance matrix
# H : measurement matrix

# x : state vector [x, y, theta, v, w]
# z : sensor measurement vector [x, y, theta, v, w]



# state model covariance matrix
Q = np.array([[params.Q_X,0,0,0,0],
                [0,params.Q_Y,0,0,0],
                [0,0,params.Q_THETA,0,0],
                [0,0,0,params.Q_V,0],
                [0,0,0,0,params.Q_W]])


# # state model matrix
def get_matrix_A(x, dt):
    A = np.array([[1,0,0,dt*np.cos(x[2]),0],
                [0,1,0,dt*np.sin(x[2]),0],
                [0,0,1,0,dt],
                [0,0,0,1,0],
                [0,0,0,0,1]])

    return A


#measurement matrix
H = np.array([[1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]])

# sensor measurement covariance matrix with camera

R_cam_on = np.array([[params.R_X,0,0,0,0],
                    [0,params.R_Y,0,0,0],
                    [0,0,params.R_THETA,0,0],
                    [0,0,0,params.R_V,0],
                    [0,0,0,0,params.R_W]])
                     



# sensor measurement covariance matrix without camera
R_cam_off = np.array([[np.inf,0,0,0,0],
                    [0,np.inf,0,0,0],
                    [0,0,np.inf,0,0],
                    [0,0,0,params.R_V,0],
                    [0,0,0,0,params.R_W]])


#compute Jacobian of the state prediction matrix

def calculate_jacobian(x, dt):
    jacobian= np.array([[1,0,-dt*x[3]*np.sin(x[2]),dt*np.cos(x[2]),0],
                        [0,1,dt*x[3]*np.cos(x[2]),dt*np.sin(x[2]),0],
                        [0,0,1,0, dt],
                        [0,0,0,1,0],
                        [0,0,0,0,1]])

    return jacobian




def ekf(dt, x_prev, P_prev, z, R):
    F= calculate_jacobian(x_prev, dt)

    # Prediction step 
    A=get_matrix_A(x_prev, dt)
    x = A @ x_prev

    # State covariance matrix prediction
    P = F @ P_prev @ F.T + Q

    # Innovation step
    y = z - H @ x

    # Innovation covariance
    S = H @ P @ H.T + R

    # Kalman gain
    K = P @ H.T @ np.linalg.inv(S)

    # Update step of state estimate
    x = x + K @ y

    # Update state covariance matrix
    P = P - K @ H @ P

    return x, P



def estimate_position( x_prev, z, P_prev, dt):
    
    #convert to numpy array for matrix operations
    x_prev=np.array(x_prev).T
    z=np.array(z).T

    #check for camera measurement validity
    if z[0] < 0 and z[1] < 0 :
        z = np.array([0,0,0,z[3],z[4]]).T
        R_cam = R_cam_off
    else:       
        R_cam = R_cam_on


    x, P = ekf(dt, x_prev, P_prev, z, R_cam)

    return x.tolist(), P
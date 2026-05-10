"""
File name: vision.py
Authors: Jin Kilidjian, Corentin Plumet, Jules Bervillé, Charles Proffit 
Created: 05/12/24
Description: camera vision functions
"""

import cv2
import numpy as np
import cv2.aruco as aruco
import parameters as params


print("OpenCV version:", cv2.__version__)
print("Aruco available:", hasattr(aruco, 'getPredefinedDictionary'))

# Define the Aruco dictionnary
aruco_dict = aruco.DICT_4X4_50
dictionary = aruco.getPredefinedDictionary(aruco_dict)
aruco_params = aruco.DetectorParameters()

# All specific parameters for the Aruco dettection
aruco_params.adaptiveThreshWinSizeMin = 3  # Minimum window size for adaptive thresholding
aruco_params.adaptiveThreshWinSizeMax = 30  # Maximum window size for adaptive thresholding
aruco_params.adaptiveThreshWinSizeStep = 1  # Step size for the window between min and max size
aruco_params.adaptiveThreshConstant = 5  # Constant subtracted from the mean in adaptive thresholding

aruco_params.minMarkerPerimeterRate = 0.03  # Minimum marker perimeter relative to image perimeter size
aruco_params.maxMarkerPerimeterRate = 0.3  # Maximum marker perimeter relative to image perimeter size

aruco_params.polygonalApproxAccuracyRate = 0.02  # Accuracy rate for polygonal curve approximation
aruco_params.minCornerDistanceRate = 0  # Minimum distance between marker corners relative to perimeter

aruco_params.minDistanceToBorder = 0  # Minimum distance between marker and image border (in pixels)

aruco_params.minMarkerDistanceRate = 0.001  # Minimum distance between detected markers, as a rate of the smaller marker's perimeter

aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # Method to refine detected corners
aruco_params.cornerRefinementWinSize = 9  # Size of the window for corner refinement
aruco_params.cornerRefinementMaxIterations = 50  # Maximum iterations for the corner refinement process
aruco_params.cornerRefinementMinAccuracy = 0.05  # Minimum accuracy for corner refinement

aruco_params.markerBorderBits = 1  # Number of bits of the marker's black border
aruco_params.perspectiveRemovePixelPerCell = 8  # Number of pixels for each cell during perspective removal
aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.1  # Margin ignored during perspective removal, relative to cell size

aruco_params.maxErroneousBitsInBorderRate = 0.4  # Maximum allowed erroneous bits in the border, as a rate
aruco_params.minOtsuStdDev = 6.0  # Minimum standard deviation for Otsu thresholding method
aruco_params.errorCorrectionRate = 0.75  # Error correction rate; used to recover bits in erroneous markers

# Global variables to be used in the different functions
goal_position = []
robot_position = []
obstacle_array = np.zeros((params.MAP_WIDTH, params.MAP_LENGTH), dtype=np.uint8) 
largest_contours = []
matrix = None  # Perspective transformation matrix
initialization = True

def get_initialization():
    return initialization

def initialize_vision(initial_frame):
    global goal_position, largest_contours, obstacle_array, matrix, initialization

    # Apply a gaussian blur on the frame
    blurred_frame = cv2.GaussianBlur(initial_frame, (5, 5), 1.25)
    
    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(blurred_frame, dictionary, parameters=aruco_params)

    if ids is None:
        cv2.aruco.drawDetectedMarkers(initial_frame, corners, ids)
        cv2.imshow("Debug", initial_frame)
        cv2.waitKey(200)
        return None,None
    
    else:
        # Create a dictionary to map marker IDs to their corner points
        corner_dict = {id_[0]: corner for id_, corner in zip(ids, corners)}

        # initialization of the map
        if all(i in corner_dict for i in [0, 1, 2, 3, 4, 5]):
            # Take the most outbond corner for each marker
            top_left = corner_dict[0][0][3]
            top_right = corner_dict[1][0][0]
            bottom_right = corner_dict[3][0][1]
            bottom_left = corner_dict[2][0][2]

            # Define source and target points for perspective transformation
            src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
            target_points = np.array([[0, 0], [params.MAP_LENGTH, 0], [params.MAP_LENGTH, params.MAP_WIDTH], [0, params.MAP_WIDTH]], dtype="float32")

            first = True

            while True:
                # Compute the perspective transform matrix
                matrix = cv2.getPerspectiveTransform(src_points, target_points)

                # Warp the perspective of the initial frame
                warped_initial_frame = cv2.warpPerspective(blurred_frame, matrix, (params.MAP_LENGTH, params.MAP_WIDTH))

                # Convert to grayscale
                gray_warped = cv2.cvtColor(warped_initial_frame, cv2.COLOR_BGR2GRAY)

                # Threshold to create a binary image where black regions are identified as obstacles
                _, obstacle_mask = cv2.threshold(gray_warped, params.THRESHOLD, 255, cv2.THRESH_BINARY_INV)

                # Find contours of the black regions (obstacles)
                contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                largest_contours = []

                # Sort the contours based on size and take the MAX_NB_OBSTACLES obstacles, this filters the markers
                largest_contours = sorted([contour for contour in contours if cv2.contourArea(contour) > 300],key=cv2.contourArea, reverse=True)[:params.MAX_NB_OBSTACLES]
                
                obstacle_array.fill(0)

                # Draw contours on the obstacle array to mark obstacle positions as 1
                for contour in largest_contours:
                    cv2.drawContours(obstacle_array, [contour], -1, 1, thickness=cv2.FILLED)

                # Draw obstacles on the warped frame
                cv2.drawContours(warped_initial_frame, largest_contours, -1, (0, 255, 0), 2)

                # Display the warped frame with detected obstacles, goal, and robot position
                if first:
                    cv2.imshow("Initialization of the map", warped_initial_frame)
                    first = False

                # Wait for a key press
                key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press

                # Function to control the threshold to refine the obstacle detection
                if key == ord('p'):  # Increase threshold
                   cv2.imshow("Initialization of the map", warped_initial_frame)
                   params.set_threshold(min(200, params.THRESHOLD + 1))
                   print(params.THRESHOLD)
                elif key == ord('m'):  # Decrease threshold
                    cv2.imshow("Initialization of the map", warped_initial_frame)
                    params.set_threshold(max(0, params.THRESHOLD - 1))
                    print(params.THRESHOLD)
                elif key == 13:  # Enter key (end adjustment)
                    break
            
            # Destroy all OpenCV windows
            cv2.destroyAllWindows()

            # Detect the goal marker
            goal_corners = corner_dict[5][0]
            goal_center_x = int(np.mean(goal_corners[:, 0]))
            goal_center_y = int(np.mean(goal_corners[:, 1]))

            # Warp the goal position
            goal_position_warped = cv2.perspectiveTransform(np.array([[[goal_center_x, goal_center_y]]], dtype='float32'), matrix)[0][0]
            goal_position = [int(goal_position_warped[0]), int(goal_position_warped[1])]
            initialization = False
            goal_to_send = [goal_position[1]*params.SIZE_CASE,goal_position[0]*params.SIZE_CASE] #in mm
            return obstacle_array.tolist(), goal_to_send

        else:
            cv2.aruco.drawDetectedMarkers(initial_frame, corners, ids)
            cv2.imshow("Debug", initial_frame)
            cv2.waitKey(200)
            return None,None


def get_robot_position(frame):
    global matrix,robot_position

    # Apply the gaussian blur
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 1.25)

    # Detect ArUco markers in the current frame
    corners, ids, _ = aruco.detectMarkers(blurred_frame, dictionary, parameters=aruco_params)

    if ids is not None:
        # Create a dictionary to map marker IDs to their corner points
        corner_dict = {id_[0]: corner for id_, corner in zip(ids, corners)}

        if 4 in corner_dict:
            robot_corners = corner_dict[4][0]

            # Calculate the center of the robot marker
            robot_center_x = int(np.mean(robot_corners[:, 0]))
            robot_center_y = int(np.mean(robot_corners[:, 1]))

            # Warp the robot position
            robot_position_warped = cv2.perspectiveTransform(np.array([[[robot_center_x, robot_center_y]]], dtype='float32'), matrix)[0][0]
            robot_center_x_warped, robot_center_y_warped = int(robot_position_warped[0]), int(robot_position_warped[1])

            # Get the top left and top right corners of the robot marker
            robot_top_left = robot_corners[0]
            robot_top_right = robot_corners[1]

            corners_to_warp = np.array([robot_top_left, robot_top_right], dtype="float32")

            # Warp the corners position
            warped_corners = cv2.perspectiveTransform(np.array([corners_to_warp], dtype="float32"), matrix)[0]
            warped_top_left = warped_corners[0]
            warped_top_right = warped_corners[1]

            # Calculate the angle of the robot
            direction_vector = np.array(warped_top_right) - np.array(warped_top_left)
            angle_rad = np.arctan2(direction_vector[1], direction_vector[0])  # y, x
            angle_rad = -angle_rad + np.pi
            robot_position = [robot_center_x_warped, robot_center_y_warped,angle_rad]

        else:
            robot_position = [-1,-1,-1]        
    
    # If no robot, return -1 to notify that it is not detected
    else:
        robot_position = [-1,-1,-1]

    robot_position_to_send = [robot_position[1]*params.SIZE_CASE,robot_position[0]*params.SIZE_CASE,robot_position[2]]
    return robot_position_to_send

# Optional display
# def display_frame(frame):
#     # Warp the frame
#     warped_frame = cv2.warpPerspective(frame, matrix, (params.MAP_LENGTH, params.MAP_WIDTH))
#     warped_frame_display = warped_frame.copy()

#     # Draw obstacles
#     cv2.drawContours(warped_frame_display, largest_contours, -1, (0, 255, 0), 2)

#     # Draw the goal position
#     if goal_position is not None:
#         cv2.circle(warped_frame_display, goal_position, 10, (0, 0, 255), -1)

#     # Draw the robot position
#     if robot_position[0] is not None and robot_position[1] is not None:
#         cv2.circle(warped_frame_display, (robot_position[0], robot_position[1]), 10, (255, 0, 0), -1)

#     # Display the warped frame with detected obstacles, goal, and robot position
#     cv2.imshow("Warped Perspective with Obstacles, Goal, and Robot", warped_frame_display)

#     # Display the current frame with the detected robot marker
#     cv2.imshow("ArUco Detection - Robot Tracking", frame)
import logging
import cv2
import numpy as np


def estimatePose(image, image_points):
    size = image.shape

    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner 
                            ])


    # Approximation of Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
                             [focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], 
                             dtype = "double"
                            )
    logging.info("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    logging.info("Rotation Vector:\n {0}".format(rotation_vector))
    logging.info("Translation Vector:\n {0}".format(translation_vector))


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    noseImagePoint = ( int(image_points[0][0]), int(image_points[0][1]))
    noseEndPointProjection2D = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return [noseImagePoint, noseEndPointProjection2D]
    

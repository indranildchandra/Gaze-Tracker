import logging
import os
import cv2
import dlib
import numpy as np
from gaze_estimator import estimateGaze

script_directory_path = os.path.dirname(os.path.realpath(__file__)) # absolute dir the script is in
shape_predictor_relative_path = "../resources/shape_predictor_68_face_landmarks.dat"
shape_predictor_path = os.path.join(script_directory_path, shape_predictor_relative_path)

#shape_predictor_path = "resources\shape_predictor_68_face_landmarks.dat"

def detectFaceLandmark(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)


    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detectedFaces = detector(img, 0)
    logging.info("Number of faces detected: {}".format(len(detectedFaces)))

    
    largestArea = 0
    if len(detectedFaces) >= 1: # Consider only the most prominent face in the frame if multiple faces are detected
        for k, d in enumerate(detectedFaces):
            length = abs(d.right() - d.left())
            breadth = abs(d.bottom() - d.top())
            area = length * breadth
            logging.info("Area of Face " + str(k+1) + ": " + str(area))
            if area > largestArea:
                faceIndex = k
                faceBB = d
                largestArea = area
        logging.info("Area of Most Prominent Face: " + str(largestArea))

        
        logging.info("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(faceIndex, faceBB.left(), faceBB.top(), faceBB.right(), faceBB.bottom()))
        # Get the landmarks/parts for the face in bounding box faceBB.
        shape = predictor(img, faceBB) # logging.info("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

        # Get Facial Landmark Points
        facialLandmarkPoints = np.empty([68, 2], dtype = "double")
        for itr in range(68):
            facialLandmarkPoints[itr][0] = shape.part(itr).x
            facialLandmarkPoints[itr][1] = shape.part(itr).y
        logging.info("Facial Landmark Points:\n" + str(facialLandmarkPoints))

        #Get Facial Landmark Features Coordinates
        facialLandmarkFeatures = np.empty([6, 2], dtype = "double")
        facialLandmarkFeatures[0][0] = facialLandmarkPoints[33][0]    # Nose tip
        facialLandmarkFeatures[0][1] = facialLandmarkPoints[33][1]    # Nose tip
        facialLandmarkFeatures[1][0] = facialLandmarkPoints[8][0]     # Chin
        facialLandmarkFeatures[1][1] = facialLandmarkPoints[8][1]     # Chin
        facialLandmarkFeatures[2][0] = facialLandmarkPoints[36][0]    # Left eye left corner
        facialLandmarkFeatures[2][1] = facialLandmarkPoints[36][1]    # Left eye left corner
        facialLandmarkFeatures[3][0] = facialLandmarkPoints[45][0]    # Right eye right corner
        facialLandmarkFeatures[3][1] = facialLandmarkPoints[45][1]    # Right eye right corner
        facialLandmarkFeatures[4][0] = facialLandmarkPoints[48][0]    # Left Mouth corner
        facialLandmarkFeatures[4][1] = facialLandmarkPoints[48][1]    # Left Mouth corner
        facialLandmarkFeatures[5][0] = facialLandmarkPoints[54][0]    # Right mouth corner
        facialLandmarkFeatures[5][1] = facialLandmarkPoints[54][1]    # Right mouth corner
        logging.info("Facial Landmark Features:\n" + str(facialLandmarkFeatures))

        # Get Pose Estimate
        [noseImagePoint, noseEndPointProjection2D] = estimateGaze(img, facialLandmarkFeatures)

        logging.info("Nose Image Point Coordinate: " + str(noseImagePoint))
        logging.info("Nose End Point Projection 2D Coordinate: " + str(noseEndPointProjection2D))

        # Overlay the Facial Landmark Points and Features on top of the Captured Frame
        for p in facialLandmarkPoints:
            cv2.circle(img, (int(p[0]), int(p[1])), 1, (255,0,0), -1)
        for p in facialLandmarkFeatures:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        # Display Pose Direction in 2D Image Plane
        cv2.line(img, noseImagePoint, noseEndPointProjection2D, (255,0,0), 2)
        # Display image
        cv2.imshow("Output with Gaze Estimation", img)

    return img
    



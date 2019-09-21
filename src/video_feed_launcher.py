import logging
import sys
import os
import glob
import cv2
import dlib
import numpy as np
from datetime import datetime
from face_landmark_detector import detectFaceLandmark

# Set these variables
frameRate = 20.0 #in Frames per Second (FPS)
frameResolution = (980, 720) #Set (980,720)) for higher resolution
test_subject_name = "test"
test_number = "1"
output_directory_path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../output"), test_subject_name), test_number) 
#Change ChildName and TestNumber with each iteration of testing

source_directory_path = os.path.dirname(os.path.realpath(__file__))
output_directory_path = os.path.abspath(output_directory_path)
if not os.path.exists(output_directory_path):
	os.makedirs(output_directory_path)

script_directory_path = os.path.dirname(os.path.realpath(__file__)) # absolute dir the script is in
shape_predictor_relative_path = "../resources/shape_predictor_68_face_landmarks.dat"
shape_predictor_path = os.path.join(script_directory_path, shape_predictor_relative_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

fileName = str(datetime.now()) #example -> '2011-05-03 17:45:35.177000'
logFileName = os.path.join(output_directory_path, ((fileName.replace("-","_").replace(" ","-").replace(":","_")).split('.'))[0]) + ".log" #example -> path/yyyy_mm_dd-hh_mm_ss.log
logging.basicConfig(filename=logFileName, level=logging.INFO)
logging.info(output_directory_path)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not(cap.isOpened()):	#Sanity Check if the Video Capture Object was initiated
	cap.open()


filePathName = os.path.join(output_directory_path, ((fileName.replace("-","_").replace(" ","-").replace(":","_")).split('.'))[0]) # example -> path/yyyy_mm_dd-hh_mm_ss
recordedFileName = filePathName + "_capture" + ".avi"
outputFileName = filePathName + "_output" + ".avi"

try:
	videoWriterObj1 = cv2.VideoWriter(recordedFileName, cv2.VideoWriter_fourcc('M','J','P','G'), frameRate, frameResolution) 
	videoWriterObj2 = cv2.VideoWriter(outputFileName, cv2.VideoWriter_fourcc('M','J','P','G'), frameRate, frameResolution)

	while cv2.waitKey(1) < 0:
		has_frame, frame = cap.read()

		if not has_frame: #(keyPressed == ord('q')) or (keyPressed == ord('Q'))
			logging.info("No frame returned...")
			logging.info("Application Terminated!")
			logging.info("----------------------------------------------------------------")
			logging.info("----------------------------------------------------------------")
			break

		videoWriterObj1.write(frame)
		outputImage = detectFaceLandmark(frame, detector, predictor)
		videoWriterObj2.write(outputImage)
		logging.info("Captured Frame and Output Image written to Video file")
		logging.info("----------------------------------------------------------------")
		logging.info("----------------------------------------------------------------")
except Exception as e:
	logging.error("An Exception occured!")
	logging.error(str(e))
finally:
	videoWriterObj1.release()
	videoWriterObj2.release()
	cap.release()
	cv2.destroyAllWindows()

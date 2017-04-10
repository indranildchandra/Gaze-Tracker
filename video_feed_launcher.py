import logging
import sys
import os
import glob
import cv2
import numpy as np
from datetime import datetime
from face_landmark_detector import detectFaceLandmark

# Set these variables
frameRate = 5.0 #in Frames per Second (FPS)
frameResolution = (640, 480) #Set (980,720)) for higher resolution
output_directory_path = os.path.join(os.path.join(os.path.join(os.getcwd(), "OutputFolder"), "ChildName"), "TestNumber") 
#Change ChildName and TestNumber with each iteration of testing

source_directory_path = os.getcwd()
output_directory_path = os.path.abspath(output_directory_path)
if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)


fileName = str(datetime.now()) #Example = '2011-05-03 17:45:35.177000'
logFileName = os.path.join(output_directory_path, ((fileName.replace("-","_").replace(" ","-").replace(":","_")).split('.'))[0]) + ".log"
logging.basicConfig(filename=logFileName, level=logging.INFO)
logging.info(output_directory_path)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not(cap.isOpened()):	#Sanity Check if the Video Capture Object was initiated
	cap.open()


filePathName = os.path.join(output_directory_path, ((fileName.replace("-","_").replace(" ","-").replace(":","_")).split('.'))[0])
recordedFileName = filePathName + "_capture" + ".avi"
outputFileName = filePathName + "_output" + ".avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  #FourCC is a 4-byte code used to specify the video codec.
videoWriterObj1 = cv2.VideoWriter(recordedFileName, -1, frameRate, frameResolution) 
videoWriterObj2 = cv2.VideoWriter(outputFileName, -1, frameRate, frameResolution)
# Frame Rate (in FPS) is equal to 5, hence we should wait 1/5 seconds = 0.2 seconds = 0.2 * 1000 milliseconds = 200 milliseconds between the consecutive frames. 
# So put waitKey(200)

while True:
	keyPressed = cv2.waitKey(int(1000/frameRate))
	if keyPressed == ord('q'):
		logging.info("Application Terminated!")
		logging.info("----------------------------------------------------------------")
		logging.info("----------------------------------------------------------------")
		break
	else:
		returnStatus, frame = cap.read()
		videoWriterObj1.write(frame)
		if returnStatus == True:
	   		outputImage = detectFaceLandmark(frame)
	   		videoWriterObj2.write(outputImage)
	   		logging.info("Captured Frame and Output Image written to Video file")
	   		logging.info("----------------------------------------------------------------")
	   		logging.info("----------------------------------------------------------------")
		else:
			logging.info("Return Status = False")


videoWriterObj1.release()
videoWriterObj2.release()
cap.release()
cv2.destroyAllWindows()

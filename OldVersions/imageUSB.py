import cv2
import time
import sys

#print("Capturing images from the camera")
#print("Press CTRL+C to terminate the program")
#counter = 0


def get_image(cam_id):
	cam = cv2.VideoCapture(cam_id)
	if cam.isOpened():
		cam.set(3,640.)
		cam.set(4,480.)
		s, img = cam.read()
		cam.release()
		return (True, cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
	else:
		cam.release()
		return (False,) 
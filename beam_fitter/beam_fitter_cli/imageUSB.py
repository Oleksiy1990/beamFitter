import cv2
import time
import sys



def get_image(cam_id):
	cam = cv2.VideoCapture(cam_id)
	if cam.isOpened():
		
		#cam.set(3,1280)
		#cam.set(4,1024)
		#cam.set(90,0)
		
		for x in range(1,5): 
			s, img = cam.read()
		#this one is the peculiar thing about these USB cams, apparently the first few frames fail
		#in fact sometimes even the later frames have glitches. This is just one unelegant 
		#way to get capture frames correctly in most cases
		
		#cam.release()
		return (True, cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)) #we turn everything into grayscale
	else:
		cam.release()
		return (False,) 
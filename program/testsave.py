import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Open!")
else:
    print("this sucks")
counter = 0

    # Capture frame-by-frame
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    counter += 1

    if counter == 5:
    	break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("framebefore1.bmp",gray)
    np.savetxt('blahblahbefore.txt',gray)
    #break
    



    
cap.release()
cv2.destroyAllWindows()
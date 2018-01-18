import cv2
import numpy as np
import os
 
device = 0
cap = cv2.VideoCapture(device)

if not cap.isOpened():
	cap.open(device)

if cap.isOpened():
	while(True):
		re, img = cap.read()

		if re:
			cv2.imshow("Frame", img)

		else:
			print("Error reading capture device")
			break

		k = cv2.waitKey(10) & 0xFF
		if k == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

else:
	print('Failed to open capture device')

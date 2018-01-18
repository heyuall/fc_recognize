# facial recognition using opencv

#Imports 
import cv2
import numpy as np
import os


face_cascade_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"data", "haarcascades", "haarcascade_frontalface_default.xml"))
eye_cascade_script_path = os.path.abspath((os.path.join(os.path.dirname(__file__),"data", "haarcascades", "haarcascade_eye.xml"))
#print("[+] Path: ", cascade_script_path)

# loading the cascade xml files for the cascade classifier
face_cascade = cv2.CascadeClassifier(face_cascade_script_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_script_path)


recognise = cv2.face.createEigenFaceRecognizer(15, 4000)  # Eigen Face recognizer is used for facial classification
recognise.load()  # load the training data here

cap = cv2.VideoCapture(0)

while(1):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #convert camera frame to read grayscale

	# -------------------------- Facial Detection Stuff Here ------------------------------------------------------
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)   #Detect faces in capture and save positions
	for (x, y, w, h) in faces:
		# ---------- This part looks for the eyes in the face for better facial recognition
		gray_face = cv2.resize((gray[y: y+h, x: x+w]),(110,110))	# isolating the face and cropping it             
		eyes = eye_cascade.detectMultiScale(gray_face)
		for (ex, ey, ew, eh) in eyes:
			ID, conf = recognise.predict(gray_face) # detect the ID of the person in the photo
			

	cv2.imshow("Eigen Facial Detection & Recognition", gray) #show the video capture

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cap.destroyAllWindows()

# import librarys:
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
import cv2 as cv

# Create values:
cap = cv.VideoCapture(1) # Change video capture

# detections:
Face_Detector = FaceDetector(minDetectionCon=10) # Change facedetector options
Hand_Detector = HandDetector(detectionCon=0.5, maxHands=2) # Change handdetector options 

# Cascades:
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml") # Read haarcascade_eye.xml
smile_cascade = cv.CascadeClassifier("haarcascade_smile.xml")# Read haarcascade_eye.xml

# Show attributes
while True:
    success, image = cap.read() # Read camera video
    Faces, img = Face_Detector.findFaces(image) # Find Faces
    Hands, image = Hand_Detector.findHands(image) # Find Hands
    eyes = eye_cascade.detectMultiScale(Faces) # Find eyes
    for (ex, ey, ew, eh) in eyes: 
        cv.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2) # rectangle eyes

    smiles = smile_cascade.detectMultiScale(Faces, 1.8, 20) # Find smiles
    for (ex, ey, ew, eh) in smiles: 
        cv.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2) # rectangle smiles

    cv.imshow("Robot vision", Faces) # Show (Faces on) camera video
    cv.imshow("Robot vision", image) # Show (Hands on) camera video
    cv.waitKey(1) # Change waitkey
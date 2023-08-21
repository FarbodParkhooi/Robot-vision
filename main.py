# import librarys:
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
import cv2 as cv

# Create values:
cap = cv.VideoCapture(0) # Change video capture

# detections:
Face_Detector = FaceDetector(minDetectionCon=10) # Change facedetector options
Hand_Detector = HandDetector(detectionCon=0.5, maxHands=2) # Change handdetector options 

# Show attributes
while True:
    success, image = cap.read() # Read camera video
    Faces, img = Face_Detector.findFaces(image) # Find Faces
    Hands, image = Hand_Detector.findHands(image) # Find Hands

    cv.imshow("Robot vision", Faces) # Show (Faces on) camera video
    cv.imshow("Robot vision", image) # Show (Hands on) camera video
    cv.waitKey(1) # Change waitkey
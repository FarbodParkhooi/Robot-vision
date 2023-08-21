# import librarys:
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
import cv2 as cv


# Create values:
cap = cv.VideoCapture(1) # Change video capture

Face_Detector = FaceDetector(minDetectionCon=10) # Change facedetector options

while True:
    success, image = cap.read() # Read camera video
    Faces, image = Face_Detector.findFaces(image)

    cv.imshow("Robot vision", Faces)
    cv.waitKey(1)
# import librarys:
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import numpy as np
import cv2 as cv

# Create values:
cap = cv.VideoCapture(1) # Change video capture
says = ["not sayed", "not sayed", "not sayed"]

# detections:
Face_Detector = FaceDetector(minDetectionCon=10) # Change facedetector options
Hand_Detector = HandDetector(detectionCon=0.5, maxHands=2) # Change handdetector options 
Pose_Detector = PoseDetector() # Change Pose detector options

# Cascades:
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml") # Read haarcascade_eye.xml
smile_cascade = cv.CascadeClassifier("haarcascade_smile.xml") # Read haarcascade_smile.xml

# defs:

""" 
This function target: 
        Resize video frame by frame
Its work?
        Yes!
"""
def resize_frame(frame, percent=150):
    width = int(frame.shape[1] * percent/ 100) # resize with
    height = int(frame.shape[0] * percent/ 100) # resize height
    dim = (width, height) # change sizes
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA) # return camera output(resized)
"""
This function target:
        Say hello
It's work?
        Yes!        
"""
def Hello():
    import pyttsx3 # import library
    engine = pyttsx3.init()
    engine.setProperty('rate', 125) # Change sepeed 
    engine.say("Hello!") # Say hello
    engine.runAndWait() # Do it!
    return "I Finished"

# Show attributes
while True:
    success, image = cap.read() # Read camera video

    # Move detection code:
    
    # # These are created for moving detector
    # rec, frame1 = cap.read() # Read camera video
    # rec, frame2 = cap.read() # Read camera video
    # # resize frames:
    # frame1 = resize_frame(frame1) # resize frame 1
    # frame2 = resize_frame(frame2) # resize frame 2
    
    # frame_diff = cv.absdiff(frame1, frame2) # frame1 - frame2
    # frame_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY) # grayscal video
    # blurred_frame = cv.GaussianBlur(frame_diff, (5,5), 1) # blur frmaes
    # _, mask = cv.threshold(blurred_frame, 10, 255, cv.THRESH_BINARY)

    # # find movments:
    # Contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # for Contour in Contours:
    #     (x, y, w, h) = cv.boundingRect(Contour)
    #     cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2) # rectangle movmed objects

    Faces, img = Face_Detector.findFaces(image) # Find Faces
    Hands, image = Hand_Detector.findHands(image) # Find Hands
    eyes = eye_cascade.detectMultiScale(Faces) # Find eyes
    smiles = smile_cascade.detectMultiScale(Faces, 1.8, 20) # Find smiles
    Poses = Pose_Detector.findPose(image) # Find Poses
    # Rectangle eyes, smiles and bodys:
    for (ex, ey, ew, eh) in eyes: 
        cv.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2) # rectangle eyes
    for (ex, ey, ew, eh) in smiles: 
        cv.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2) # rectangle smiles
    lmList, bboxInfo = Pose_Detector.findPosition(image, bboxWithHands=False)
    if bboxInfo:
        center = bboxInfo["center"]
        cv.circle(image, center, 5, (255, 0, 255), cv.FILLED)
    
    # Say words after:
    if says[0] != "sayed":
        if len(Poses) > 0: # if body is in image then:
            if Hands: # if hand is in camera
                if Faces in image: # if face is in camera
                    if len(eyes) > 0: # if eye is in camera
                        code = Hello() # Say hello
                        says[0] = "sayed" # add sayed
    else:
        pass

        """
        bad az inke How_are_you() anjam shod age fasele beyn angosht shast va eshare kamtar az {x} bood kr masalan bege khoobam bege "Its great!"
        """

    # Show it!
    cv.imshow("Robot vision", Faces) # Show (Faces on) camera video
    cv.imshow("Robot vision", image) # Show (Hands on) camera video
    if cv.waitKey(1) & 0xFF == ord('q'): # If q pressed then:
        break
cap.release()
cv.destroyAllWindows()
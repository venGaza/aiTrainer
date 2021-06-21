import cv2
from cv2 import cv2
import mediapipe as mp
import time
from math import atan2, degrees

from numpy.lib.function_base import angle


class PoseDetector():
    """API for MediaPipe Pose.

    MediaPipe Pose processes an RGB image and returns pose landmarks on the 
    most prominent person detected.

    Please refer to https://solutions.mediapipe.dev/pose#python-solution-api 
    for usage examples.
    """

    def __init__(self, mode=False, upperBody=False, 
                 smooth=True, detectConf=0.5, trackConf=0.5):
        self.mode = mode
        self.upperBody = False
        self.smooth = True
        self.detectConf = detectConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.mpPose = mp.solutions.mediapipe.python.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smooth, 
                                     self.detectConf, self.trackConf)


    def findPose(self, img, draw=True):
        """Finds the pose landmarks in the image.

        Args:
            img: An RGB image to use to find pose landmarks.
            draw: Bool value determines if landmarks are drawn on image.

        Returns:
            An image with landmarks drawn(if specified) by MediaPipe.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.pose.process(imgRGB)

        # Adds the landmarks to the image if detected
        if draw and self.results.pose_landmarks:
            imgRGB.flags.writeable = True
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
                                       self.mpPose.POSE_CONNECTIONS)

        return img


    def findPosition(self, img, draw=True):
        """Converts MediaPipe landmarks into coords and uses openCV draw.

        Args:
            img: An RGB image to use to find pose landmarks.
            draw: Bool value determines if landmarks are drawn on image.

        Returns:
             The x,y coordinates of the pose landmarks.
        """
        self.lmList = [[0,0,0]] * 33 # Create list container for landmarks
        h, w, _ = img.shape

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList[id] = [id, cx, cy]
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True, absolute=True) -> int:
        """Finds the angle between a set of points (p2 is the anchor).

        Args:
            img: An RGB image to use to draw landmarks.
            p1: An integer representing the landmark id of point 1.
            p2: An integer representing the landmark id of point 2.
            p3: An integer representing the landmark id of point 3.
            draw: Bool flag for drawing landmarks.
            absolute: Bool flag to reduce interval to positive integers

        Returns:
             Returns the angle (int) of the points.
        """
        # Get coordinates for each point
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Find the angle for the joint
        angle = degrees(atan2(y3-y2, x3-x2) - atan2(y1-y2, x1-x2))
        angle =  angle % 360 # Reduce angle 
        angle = (angle + 360) % 360  # Force pos remainder (0 <= angle < 360)
        if (angle > 180):  # Force min abs val residue (-180 < angle <= 180)
            angle -= 360
        if absolute: # Reduce to positive interval
            angle = int(abs(angle))
        
        # Draw circle at each point, connect them, display joint angle
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 2)
            cv2.circle(img, (x1,y1), 5, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x1,y1), 10, (0,255,0), 2)
            cv2.circle(img, (x2,y2), 5, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x2,y2), 10, (0,255,0), 2)
            cv2.circle(img, (x3,y3), 5, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x3,y3), 10, (0,255,0), 2)
            if p2 & 1:
                cv2.putText(img, f'{angle}', (x2+40, y2+40), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            else:
                cv2.putText(img, f'{angle}', (x2-80, y2+40), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return angle

def main():
    pTime = 0
    cap = cv2.VideoCapture('./poseVideos/basketball.mp4')
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if success:
            img = detector.findPose(img)
            lmList = detector.findPosition(img)
            if lmList:
                print(lmList[14])

            # Output frames per second
            cTime = time.time()
            fps = int(1 / (cTime - pTime))
            pTime = cTime
            cv2.putText(img, str(fps), (70,50), cv2.FONT_HERSHEY_PLAIN, 
            3, (255, 0, 0), 3)

            # Show image with pose drawing
            cv2.imshow('Image', img)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
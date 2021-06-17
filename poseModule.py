from cv2 import cv2
import mediapipe as mp
import time


class PoseDetector():
    """API for MediaPipe Pose.

    MediaPipe Pose processes an RGB image and returns pose landmarks on the most
    prominent person detected.

    Please refer to https://solutions.mediapipe.dev/pose#python-solution-api for
    usage examples.
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
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # Adds the landmarks to the image if detected
        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
                                       self.mpPose.POSE_CONNECTIONS)

        return img


    def findPosition(self, img, draw=True):
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        
        return lmList


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

                # Track a specific landmark
                #cv2.circle(img, (lmList[14][1],lmList[14[2]]), 5, (255,0,0), cv2.FILLED)

            # Output frames per second
            cTime = time.time()
            fps = int(1 / (cTime - pTime))
            pTime = cTime
            cv2.putText(img, str(fps), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Show image with pose drawing
            cv2.imshow('Image', img)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
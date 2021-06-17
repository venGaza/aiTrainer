from cv2 import cv2
import mediapipe as mp
import poseModule as pm
import numpy as np
import os
import time as time

CAM_WIDTH, CAM_HEIGHT = 640, 480


def main(record=False):
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    cap = cv2.VideoCapture('/videos/curls.mp4')
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    detector = pm.PoseDetector(detectConf=0.75)
    previousTime = 0

    while True:
        img, success = cap.read()

        if success:
            img = detector.findPose(img)
            landmark_list = detector.findPosition(img, draw=False)

            if landmark_list:
                pass

            # Output FPS
            currentTime = time.time()
            fps = int(1 / (currentTime-previousTime))
            previousTime = currentTime
            cv2.putText(img, f'FPS: {fps}', (20, 30), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 0), 1)

            # Output image
            if record: 
                out.write(img)
            cv2.imshow('Image', img)
            key = cv2.waitKey(1)
            if key == 27:
                break

if __name__ == '__main__':
    main()



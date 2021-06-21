import cv2
from cv2 import cv2 # Language interpreter workaround
import mediapipe as mp
import poseModule as pm
import numpy as np
import os
import time as time

# TODO:
# Z-rotation of landmark triangles to normalize angle from 3d to 2d space
# Responsive dashboard info for different resolutions
# Add more exercises
# Visual indicator of rep completion

CAM_WIDTH, CAM_HEIGHT = 640, 480
BLUE, GREEN, RED = (255,0,0), (0,255,0), (0,0,255)
WHITE, BLACK = (255,255,255), (0,0,0)

"""
Example
https://www.pexels.com/collections/sport-3sgmbq9/
"""

"""Store the state of an exercise"""
class Exercise:
    def __init__(self, exercise_name):
        self.name = exercise_name
        self.attempt = False
        self.repetition_count = 0


def evaluate_squat_clean_jerk(detector, img, ex) -> None:
    """Counts repetitions for the squat clean and jerk exercise.

    Args:
        detector: A string representing the path to the folder.
        img: An image to draw relevant landmarks. 
        ex: An Exercise object which stores the state.

    Useful landmarks:
        Curls: Left Arm (11, 13, 15) Right Arm (12, 14, 16)
        Squats: Left Leg (23, 25, 27) Right Leg (24, 26, 28)
    """
    left_leg_angle = detector.findAngle(img, 23, 25, 29)
    right_leg_angle = detector.findAngle(img, 24, 26, 30)
    left_arm_angle = detector.findAngle(img, 11, 13, 15)
    right_arm_angle = detector.findAngle(img, 12, 14, 16)

    # Check if athlete has reset pose for next attempt
    if right_arm_angle < 90 and right_leg_angle < 120:
        ex.attempt = True
    
    # Check for repetition
    if ex.attempt:
        if max(left_leg_angle, right_leg_angle) > 170:
            if min(right_arm_angle, left_arm_angle) > 140:
                ex.repetition_count += 1
                ex.attempt = False


def main(record=False, exercise=''):
    ex = Exercise(exercise)
    cap = cv2.VideoCapture('./videos/crossfit2.mov')
    # cap.set(3, CAM_WIDTH)
    # cap.set(4, CAM_HEIGHT)
    
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height =int(cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    detector = pm.PoseDetector(detectConf=0.5)
    previousTime = 0

    while True:
        success, img = cap.read()
        # img = cv2.resize(img, (640, 480))

        if success:
            img = detector.findPose(img, draw=False)
            if detector.findPosition(img, draw=False):
                evaluate_squat_clean_jerk(detector, img, ex)
                
        # Output dashboard (FPS / Mode / Exercise / Repetitions)
        currentTime = time.time()
        fps = int(1 / (currentTime-previousTime))
        previousTime = currentTime
        cv2.putText(img, f'FPS: {fps}', (20, 30), 
                    cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)
        cv2.putText(img, f'MODE: DEGREES', (20, 50), 
                    cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)
        cv2.putText(img, f'EXERCISE: {exercise.upper()}', (20, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)
        cv2.putText(img, f'{ex.repetition_count}', (40, 200), 
                    cv2.FONT_HERSHEY_PLAIN, 10, GREEN, 5)
        
        # Display image
        if record: 
            out.write(img)
        cv2.imshow('Exercise Pose', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    
    cap.release()
    if record: 
        out.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main(record=False, exercise='Squat Clean and Jerk')
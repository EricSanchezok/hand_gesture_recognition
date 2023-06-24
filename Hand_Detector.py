import mediapipe as mp
import cv2
import numpy as np


class mediaPipe_Hand_Detector:
    def __init__(self, static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.1) -> None:
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode, max_num_hands, model_complexity, min_detection_confidence, min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=5)

    def get_landmarks(self, color_image, draw_fingers=True):

        results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks and draw_fingers:
            for handLms in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(color_image, handLms, self.mpHands.HAND_CONNECTIONS, self.handLmsStyle, self.handConStyle)

        if results.multi_hand_world_landmarks and results.multi_hand_landmarks:
            handLandmarks_points = []
            handworldLandmarks_points = [] 

            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    handLandmarks_points.append(lm.x)
                    handLandmarks_points.append(lm.y)
                    handLandmarks_points.append(lm.z)

            handLandmarks_points = np.array(handLandmarks_points).reshape(1, -1)

            for handLms in results.multi_hand_world_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    handworldLandmarks_points.append(lm.x)
                    handworldLandmarks_points.append(lm.y)
                    handworldLandmarks_points.append(lm.z)

            handworldLandmarks_points = np.array(handworldLandmarks_points).reshape(1, -1)

        else:
            handworldLandmarks_points = None
            handLandmarks_points = None

        return handLandmarks_points, handworldLandmarks_points
import mediapipe as mp
import cv2
import numpy as np
import os



class DataSaver:
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.file = open(self.file_name, 'a')
        self.save_num = 0
        self.save_label = 0
        self.start_save = False


    def writeData(self, results, label:int):
        if results is not None:
            if results.multi_hand_world_landmarks:
                for handLms in results.multi_hand_world_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        self.file.write(lm.x.__str__() + "," + lm.y.__str__() + "," + lm.z.__str__() + ",")
                    self.file.write(label.__str__() + "\n")
                print("label: " + label.__str__() + " is written")


    def readytosave(self, key, results, flip):

        if key & 0xFF == ord('k'):
            # 取反
            self.start_save = not self.start_save
            self.save_num = 0

        # #握拳状态
        # if key & 0xFF == ord('0'):
        #     self.save_label = 0    
        # #伸出食指状态
        # if key & 0xFF == ord('1'):
        #     self.save_label = 1
        # #OK状态
        # if key & 0xFF == ord('2'):
        #     self.save_label = 2
        # #全手掌打开状态
        # if key & 0xFF == ord('3'):
        #     self.save_label = 3

        #0状态
        if key & 0xFF == ord('0'):
            self.save_label = 0    
        #1状态
        if key & 0xFF == ord('1'):
            self.save_label = 1
        #2状态
        if key & 0xFF == ord('2'):
            self.save_label = 2
        #3状态
        if key & 0xFF == ord('3'):
            self.save_label = 3
        #4状态
        if key & 0xFF == ord('4'):
            self.save_label = 4
        #5状态
        if key & 0xFF == ord('5'):
            self.save_label = 5
        #6状态
        if key & 0xFF == ord('6'):
            self.save_label = 6
        #7状态
        if key & 0xFF == ord('7'):
            self.save_label = 7
        #8状态
        if key & 0xFF == ord('8'):
            self.save_label = 8
        #9状态
        if key & 0xFF == ord('9'):
            self.save_label = 9
        

        print("保存状态:", self.start_save, "保存编号:", self.save_label, "保存次数:", self.save_num)
        if self.start_save:
            self.writeData(results, self.save_label)
            self.save_num += 1

     

class mediaPipeHand:
    def __init__(self, static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.1) -> None:
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode, max_num_hands, model_complexity, min_detection_confidence, min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=5)

    def get_world_points(self, color_image, drawPoints=True):

        results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks and drawPoints:
            for handLms in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(color_image, handLms, self.mpHands.HAND_CONNECTIONS, self.handLmsStyle, self.handConStyle)

        if results.multi_hand_world_landmarks:
            points = []
            for handLms in results.multi_hand_world_landmarks:

                for id, lm in enumerate(handLms.landmark):
                    points.append(lm.x)
                    points.append(lm.y)
                    points.append(lm.z)

            points = np.array(points).reshape(1, -1)
        else:
            points = None
            results = None
        
        return points, results
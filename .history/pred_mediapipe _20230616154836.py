import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time

import torch.nn as nn
import torch
import pandas as pd
import data_process as dp

from pointnet_model import get_model

file_name = "test_data.csv"


class data_saver:
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.file = open(self.file_name, 'a')
        self.save_num = 0
        self.save_label = 0
        self.start_save = False


    def writeData(self, results, label:int):
        if results.multi_hand_world_landmarks:
            for handLms in results.multi_hand_world_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    self.file.write(lm.x.__str__() + "," + lm.y.__str__() + "," + lm.z.__str__() + ",")
                self.file.write(label.__str__() + "\n")
            print("label: " + label.__str__() + " is written")
        else:
            print("No hand detected")
            

    def keyPress(self, key, results, saveModel=False):
        if saveModel:
            if key & 0xFF == ord('k'):
                # 取反
                self.start_save = not self.start_save
                self.save_num = 0
            #握拳状态
            if key & 0xFF == ord('0'):
                self.save_label = 0    
            #伸出食指状态
            if key & 0xFF == ord('1'):
                self.save_label = 1
            #OK状态
            if key & 0xFF == ord('2'):
                self.save_label = 2
            #全手掌打开状态
            if key & 0xFF == ord('3'):
                self.save_label = 3
            print("保存状态:", self.start_save, "保存编号:", self.save_label, "保存次数:", self.save_num)
            if self.start_save:
                self.writeData(results, self.save_label)
                self.save_num += 1


def writeData(results, label:int):
    if results.multi_hand_world_landmarks:
        for handLms in results.multi_hand_world_landmarks:
            with open(file_name, 'a') as f:
                for id, lm in enumerate(handLms.landmark):
                    f.write(lm.x.__str__() + "," + lm.y.__str__() + "," + lm.z.__str__() + ",")
                f.write(label.__str__() + "\n")
        print("label: " + label.__str__() + " is written")
    else:
        print("No hand detected")

        
    
# 加载模型
model = get_model(num_classes=4, global_feat=True, feature_transform=False, channel=3)
model.load_state_dict(torch.load('model.pth'))



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
            print("No hand detected")
            results = None
            points = None
            

        return results, points
    
class FPS:
    def __init__(self) -> None:
        self.pTime = 0
        self.cTime = 0

    def showFPS(self, img):
        self.cTime = time.time()
        fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)
    


pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

start_save = False
save_count = 0
save_label = 0
save_num = 0

hand_detector = mediaPipeHand()
fps = FPS()

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 1024x768 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 1920x1080 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #深度小于0的转化为灰色
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        depth_image = np.where((depth_image_3d <= 0), grey_color, color_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        results, points = hand_detector.get_world_points(color_image, drawPoints=True)
        
        if points is not None:

            X, _ = dp.data_to_points_cloud(points)

            with torch.no_grad():
                model.eval()
                predictions, _, _ = model(X)
                #predictions = predictions.max(1)[1]
                print(predictions)

        fps.showFPS(color_image)
                    
        images = np.hstack((color_image, depth_image))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break









                        
finally:
    pipeline.stop()
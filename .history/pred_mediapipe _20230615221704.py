import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time


import torch.nn as nn
import torch

import pandas as pd

import data_process as dp

file_name = "train_data.csv"


def writeData(results, label:int):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            with open(file_name, 'a') as f:
                for id, lm in enumerate(handLms.landmark):
                    f.write(lm.x.__str__() + "," + lm.y.__str__() + "," + lm.z.__str__() + ",")
                f.write(label.__str__() + "\n")
        print("label: " + label.__str__() + " is written")
    else:
        print("No hand detected")

class MLP(nn.Module):
    def __init__(self, in_features, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)
    
# 加载模型
model = MLP(63, 0.1)  # 创建一个与训练时相同结构的模型对象
model.load_state_dict(torch.load('model.pth'))


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.1)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0,255,0), thickness=5)


pTime = 0
cTime = 0


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

        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        imgHeight, imgWidth, _ = color_image.shape
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(color_image, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                points = []
                
                for id, lm in enumerate(handLms.landmark):
                    points.append(lm.x)
                    points.append(lm.y)
                    points.append(lm.z)

            points = np.array(points).reshape(1, -1)
            
            X, _ = dp.data_processing(points)
            
            with torch.no_grad():
                model.eval()
                predictions = model(X)
                print(predictions)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(color_image, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)
                    

        images = np.hstack((color_image, depth_image))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        #竖大拇指
        if key & 0xFF == ord('0'):
            writeData(results, 0)
        #按住大拇指&握拳
        if key & 0xFF == ord('1'):
            writeData(results, 1)
        #伸出食指
        if key & 0xFF == ord('2'):
            writeData(results, 2)
        #比ok
        if key & 0xFF == ord('3'):
            writeData(results, 3)

                        
finally:
    pipeline.stop()
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation

from filterpy.kalman import KalmanFilter

import point_process as pp

file_name = "world_data.csv"


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


    def readytosave(self, key, results, saveMode=False):
        if saveMode:
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

        return self.cTime - self.pTime
    
class realSense:
    def __init__(self) -> None:
        self.pipeline = rs.pipeline()

        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.cfg = self.pipeline.start(self.config)

        self.depth_sensor = self.cfg.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.profile_depth = self.cfg.get_stream(rs.stream.depth)
        self.profile_color = self.cfg.get_stream(rs.stream.color)

        self.extrinsics = self.profile_color.get_extrinsics_to(self.profile_depth)

        self.align = rs.align(rs.stream.color)

    def get_frame(self):

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        self.camera_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        return aligned_depth_frame, color_frame
    


hand_detector = mediaPipeHand()
fps = FPS()
data_saver = DataSaver(file_name)

L515 = realSense()



saveMode = False



try:
    while True:
        aligned_depth_frame, color_frame = L515.get_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        ori_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        ori_color_image = np.asanyarray(color_frame.get_data())

        show_depth_image = ori_depth_image.copy()
        show_color_image = ori_color_image.copy()

        #深度小于0的转化为灰色
        grey_color = 153
        show_depth_image_3d = np.dstack((show_depth_image,show_depth_image,show_depth_image)) #depth image is 1 channel, color is 3 channels
        show_depth_image = np.where((show_depth_image_3d <= 0), grey_color, show_color_image)

        results, points = hand_detector.get_world_points(show_color_image, drawPoints=True)

        dt = fps.showFPS(show_color_image)

        if points is not None and not saveMode:

            fliter_point = pp.point_proccessing(points, results, ori_color_image, aligned_depth_frame, L515, dt)
            print(fliter_point)
                    
        images = np.hstack((show_color_image, show_depth_image))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break



        data_saver.readytosave(key, results, saveMode)





                        
finally:
    L515.pipeline.stop()
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

file_name = "test_data.csv"


class DataSaver:
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
            #print("No hand detected")
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
    


pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

stream_profile = profile.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
camera_intrinsics = stream_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

align = rs.align(rs.stream.color)


hand_detector = mediaPipeHand()
fps = FPS()
data_saver = DataSaver(file_name)

# 加载模型
model = get_model(num_classes=4, global_feat=True, feature_transform=False, channel=3)
model.load_state_dict(torch.load('model.pth'))

saveMode = False

# 创建一个空的图形对象
fig = plt.figure(figsize=(24, 16))

# 创建一个空的坐标轴对象
ax = plt.axes(projection='3d')
ax.view_init(elev=15, azim=0)


# 初始化数据
x = []
y = []
z = []

midy = -0.55
midz = -0.16
length = 2.0

# 设置坐标轴范围
x_min, x_max = 0, length
y_min, y_max = midy - length/2, midy + length/2
z_min, z_max = midz - length/2, midz + length/2

def plot_update(depth_point):
    # 在更新函数中修改数据
    x.append(depth_point[0])
    y.append(depth_point[1])
    z.append(depth_point[2])

    # 清空坐标轴
    ax.clear()

    # 绘制新的数据点
    ax.scatter3D(x, y, z)

    # 设置坐标轴范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # 添加其他绘图元素
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand Path')

ani = animation.FuncAnimation(fig, plot_update, interval=30)

# 初始化卡尔曼滤波器
kf = KalmanFilter(dim_x=6, dim_z=3)  # 状态向量维度为3，观测向量维度为3

# 定义观测矩阵
kf.H = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0]])

q = 0.001  # 过程噪声方差
# 定义过程噪声协方差矩阵
kf.Q = np.eye(6) * q  # q为过程噪声方差

r = 0.1  # 测量噪声方差
# 定义测量噪声协方差矩阵
kf.R = np.eye(3) * r  # r为测量噪声方差

# 初始化状态向量和状态协方差矩阵
kf.P = np.eye(6)

filter_init = False


try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 1024x768 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 1920x1080 depth image
        #color_frame = aligned_frames.get_color_frame()

        color_frame = aligned_frames.first(rs.stream.color)

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

        depth_point = None
        
        if points is not None and not saveMode:

            X, _ = dp.data_to_points_cloud(points)

            with torch.no_grad():
                model.eval()
                predictions, _, _ = model(X)
                pred_val = predictions.max(1)[0]
                pred_index = predictions.max(1)[1]
                #print(pred_val, pred_index, end='\r')

            img_width, img_height = color_image.shape[1], color_image.shape[0]

            if pred_val >= 0.8 and pred_index <= 1.2 and pred_index == 1:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        if id == 8:
                            cx, cy = min(int(lm.x * img_width), img_width-1), min(int(lm.y * img_height), img_height-1)
                            cv2.circle(color_image, (cx, cy), 30, (255, 0, 0), cv2.FILLED)
                            depth_value = aligned_depth_frame.get_distance(cx, cy)
                            depth_pixel = [cx, cy]
                            depth_point = rs.rs2_deproject_pixel_to_point(camera_intrinsics, depth_pixel, depth_value)

                            #判断是否是0
                            if depth_point[0] == 0 and depth_point[1] == 0 and depth_point[2] == 0:
                                depth_point = None
                                continue
                            #判断是否大于2
                            if depth_point[0] > 2:
                                depth_point = None
                                continue

                            mid_point = np.array([depth_point[2], -depth_point[0], -depth_point[1]], dtype=np.float32)

                            depth_point = mid_point

                            if not filter_init:
                                kf.x = np.array([[depth_point[0]], [depth_point[1]], [depth_point[2]], [0], [0], [0]], dtype=np.float32)
                                filter_init = True

                            
                            #print(depth_point)




        dt = fps.showFPS(color_image)

        #假设目标的运动模式为匀速运动，因此状态转移矩阵为：
        kf.F = np.array([[1, 0, 0, dt, 0, 0],
                        [0, 1, 0, 0, dt, 0],
                        [0, 0, 1, 0, 0, dt],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])
        
        if depth_point is not None:
        
            # 预测步骤
            kf.predict()

            measurement = np.array([depth_point[0], depth_point[1], depth_point[2]])

            # 更新步骤
            kf.update(measurement)

            fliter_point = kf.x[:3, 0]


            print("fuckyou!", fliter_point, type(fliter_point), fliter_point.shape)

        #     plot_update(fliter_point)

        # plt.pause(0.01)
                    
        images = np.hstack((color_image, depth_image))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break



        data_saver.readytosave(key, results, saveMode)





                        
finally:
    pipeline.stop()
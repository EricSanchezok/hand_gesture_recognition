import torch.nn as nn
import torch
import pandas as pd
import data_process as dp

import pyrealsense2 as rs
import numpy as np

from pointnet_model import get_model
from filterpy.kalman import KalmanFilter

import cv2

# 加载模型
model = get_model(num_classes=4, global_feat=True, feature_transform=False, channel=3)
model.load_state_dict(torch.load('model.pth'))


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

def get_pred(points):

    X, _ = dp.data_to_points_cloud(points)

    with torch.no_grad():
        model.eval()
        predictions, _, _ = model(X)
        pred_val = predictions.max(1)[0]
        pred_index = predictions.max(1)[1]
        #print(pred_val, pred_index, end='\r')

    return pred_val, pred_index


def get_ori_depth(pred_val, pred_index, results, color_image, aligned_depth_frame, camera_intrinsics):

    img_width, img_height = color_image.shape[1], color_image.shape[0]

    if pred_index == 1:
        for handLms in results.multi_hand_landmarks:
            nodes = []
            depth_point, depth_pixel = None, None
            for id, lm in enumerate(handLms.landmark):
                cx, cy = min(int(lm.x * img_width), img_width-1), min(int(lm.y * img_height), img_height-1)
                nodes.append([cx, cy])
                if id == 8:
                    # 画出手指的关节点
                    cv2.circle(color_image, (cx, cy), 20, (255, 0, 0), cv2.FILLED)

                    cx = min(cx, img_width-1)
                    cy = min(cy, img_height-1)
                    cx = max(cx, 1)
                    cy = max(cy, 1)

                    # 获取深度图中的深度值
                    depth_value = aligned_depth_frame.get_distance(cx, cy)
                    depth_pixel = [cx, cy]
                    depth_point = rs.rs2_deproject_pixel_to_point(camera_intrinsics, depth_pixel, depth_value)

                    mid_point = np.array([depth_point[2], -depth_point[0], -depth_point[1]], dtype=np.float32)

                    depth_point = mid_point

            # 计算中心点
            nodes = np.array(nodes)
            center = np.mean(nodes, axis=0)

            return depth_point, depth_pixel, center
    
    return None, None

def get_process_depth(depth_point, depth_pixel, aligned_depth_frame, camera_intrinsics, dt):

    #判断是否是0
    if depth_point.any() == 0 or depth_point[0] >= 2:
        depth_points = []
        imgWidth, imgHeight = aligned_depth_frame.get_width(), aligned_depth_frame.get_height()
        #在该点附近寻找
        for i in range(-3, 3):
            for j in range(-3, 3):
                x_new = min(depth_pixel[0]+i, imgWidth-1)
                x_new = max(x_new, 1)
                y_new = min(depth_pixel[1]+j, imgHeight-1)
                y_new = max(y_new, 1)
                depth_pixel = [x_new, y_new]
                depth_value = aligned_depth_frame.get_distance(depth_pixel[0], depth_pixel[1])
                depth_point = rs.rs2_deproject_pixel_to_point(camera_intrinsics, depth_pixel, depth_value)
                if depth_point[0] != 0 and depth_point[0] < 2:
                    depth_points.append(depth_point)
        
        if len(depth_points) == 0:
            return None
        else:
            #均值滤波
            depth_points = np.array(depth_points)
            depth_point = np.mean(depth_points, axis=0)

    global filter_init

    if not filter_init:
        kf.x = np.array([[depth_point[0]], [depth_point[1]], [depth_point[2]], [0], [0], [0]], dtype=np.float32)
        filter_init = True
        

    #假设目标的运动模式为匀速运动，因此状态转移矩阵为：
    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])
    
    # 预测步骤
    kf.predict()

    print(depth_point)

    measurement = np.array([depth_point[0], depth_point[1], depth_point[2]])

    # 更新步骤
    kf.update(measurement)

    fliter_point = kf.x[:3, 0]

    print("fuckyou!", fliter_point, type(fliter_point), fliter_point.shape)

    return fliter_point





def point_proccessing(points, results, color_image, aligned_depth_frame, camera_intrinsics, dt):

    pred_val, pred_index = get_pred(points)

    depth_point, depth_pixel = get_ori_depth(pred_val, pred_index, results, color_image, aligned_depth_frame, camera_intrinsics)

    if depth_point is not None and depth_pixel is not None:

        fliter_point = get_process_depth(depth_point, depth_pixel, aligned_depth_frame, camera_intrinsics, dt)

        return fliter_point

    else:
        return None

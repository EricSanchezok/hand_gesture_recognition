import torch.nn as nn
import torch
import pandas as pd
import data_process as dp

import pyrealsense2 as rs
import numpy as np

from pointnet_model import get_model

import cv2


# 加载模型
model = get_model(num_classes=4, global_feat=True, feature_transform=False, channel=3)
model.load_state_dict(torch.load('model.pth'))

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
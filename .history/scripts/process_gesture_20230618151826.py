import torch
from pointnet_model import get_model
import data_process as dp

import os

import numpy as np
import cv2


# 加载模型
model = get_model(num_classes=10, global_feat=True, feature_transform=False, channel=3)

model_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'model/model.pth')

model.load_state_dict(torch.load(model_dir))

history_index_list = []

def get_pred(points, angle_list, IsPrint=False):

    if points is None:
        return None, None

    X, _ = dp.data_to_points_cloud(points)

    with torch.no_grad():
        model.eval()
        predictions, _, _ = model(X)
        ori_pred_val = predictions.max(1)[0]
        ori_pred_index = predictions.max(1)[1]

        pred_val = ori_pred_val.item()
        pred_index = ori_pred_index.item()

    if pred_index == 1 or pred_index == 9:
        if angle_list[0] > 165:
            pred_index = 1
        else:
            pred_index = 9


    most_index = history_most_index(pred_index, history_index_list)

    if IsPrint:
        print(pred_val, most_index)

    return most_index


def history_most_index(index, history_index_list):

    history_index_list.append(index)
    if len(history_index_list) > 5:
        history_index_list.pop(0)

    # 选取list中出现次数最多的元素
    most_index = max(history_index_list, key=history_index_list.count)

    return most_index


def cal_finger_angle(results, show_color_image):

    if results is not None:
        #食指、中指、无名指、小手指
        angle_list = []
        joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]  # 手指关节序列

        LM = results.multi_hand_landmarks[0]
        for joint in joint_list:
            a = np.array([LM.landmark[joint[0]].x, LM.landmark[joint[0]].y])
            b = np.array([LM.landmark[joint[1]].x, LM.landmark[joint[1]].y])
            c = np.array([LM.landmark[joint[2]].x, LM.landmark[joint[2]].y])

            # 计算弧度
            radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

            if angle > 180.0:
                angle = 360 - angle

            angle_list.append(angle)

            cv2.putText(show_color_image, str(round(angle, 2)), tuple(np.multiply(b, [1280, 720]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
        return angle_list
    
    return None
import torch
from pointnet_model import get_model
import data_process as dp

import os

import numpy as np
import cv2


# 加载模型
model = get_model(num_classes=4, global_feat=True, feature_transform=False, channel=3)

model_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'model/model.pth')

model.load_state_dict(torch.load(model_dir))


def get_pred(points):

    if points is None:
        return None, None

    X, _ = dp.data_to_points_cloud(points)

    with torch.no_grad():
        model.eval()
        predictions, _, _ = model(X)
        pred_val = predictions.max(1)[0]
        pred_index = predictions.max(1)[1]

    return pred_val, pred_index

def cal_indexfinger_angle(results, show_color_image):

    if results is not None:

        joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]  # 手指关节序列
        if results.multi_hand_world_landmarks:
            LM = results.multi_hand_world_landmarks[0]
            for joint in joint_list:
                a = np.array([LM.landmark[joint[0]].x, LM.landmark[joint[0]].y])
                b = np.array([LM.landmark[joint[1]].x, LM.landmark[joint[1]].y])
                c = np.array([LM.landmark[joint[2]].x, LM.landmark[joint[2]].y])

                # 计算弧度
                radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

                if angle > 180.0:
                    angle = 360 - angle

                cv2.putText(show_color_image, str(round(angle, 2)), tuple(np.multiply(b, [1280, 720]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
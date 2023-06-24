import torch

import hand_roscv.process_data as pd

import numpy as np
import cv2

history_index_list = []

def get_pred(server):

    if server.handLandmarks_points is None:
        return None, None

    X, _ = pd.data_to_points_cloud(server.handLandmarks_points)

    with torch.no_grad():
        server.model.eval()
        predictions, _, _ = server.model(X)
        ori_pred_val = predictions.max(1)[0]
        ori_pred_index = predictions.max(1)[1]

        pred_val = ori_pred_val.item()
        pred_index = ori_pred_index.item()

    if pred_index == 1 or pred_index == 9:
        if server.angle_list[0] > 165:
            pred_index = 1
        else:
            pred_index = 9


    most_index = history_most_index(pred_index, history_index_list)

    if server.print_pred_index:
        print(pred_val, most_index)

    server.pred_index = most_index


def history_most_index(index, history_index_list):

    history_index_list.append(index)
    if len(history_index_list) > 5:
        history_index_list.pop(0)

    # 选取list中出现次数最多的元素
    most_index = max(history_index_list, key=history_index_list.count)

    return most_index


def cal_finger_angle(server):

    if server.results is not None:
        #食指、中指、无名指、小手指
        angle_list = []
        joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]  # 手指关节序列

        LM = server.results.multi_hand_landmarks[0]
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

            cv2.putText(server.show_color_image, str(round(angle, 2)), tuple(np.multiply(b, [1280, 720]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
        server.angle_list = angle_list
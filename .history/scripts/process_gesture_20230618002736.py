import torch
from pointnet_model import get_model
import data_process as dp

import os


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

def cal_tip_angle(results):
    joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]  # 手指关节序列
    if results.right_hand_landmarks:
        RHL = results.right_hand_landmarks
        for joint in joint_list:
            a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
            b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
            c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
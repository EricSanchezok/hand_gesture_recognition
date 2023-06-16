# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import mediapipe as mp
import time

import torch.nn as nn
import torch

import pandas as pd

def normalize(tensor):
    """
    标准化张量
    """
    # 计算每行的均值和标准差
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)

    # 标准化张量
    normalized_tensor = (tensor - mean) / std


    return normalized_tensor

def data_processing(data):

    if type(data) == pd.DataFrame:
        #将所有object转换为float
        data = data.astype(float)

        num_columns = data.shape[1]

        #给dataframe加上列名
        column_names = []
        for i in range(num_columns-1):
            column_names.append(f'{i//3}{"xyz"[i%3]}')
        column_names.append('label')

        data.columns = column_names

        # 进行独热编码
        one_hot_encoded = pd.get_dummies(data['label'], prefix='label')

        # 将编码后的列与原数据合并
        data = pd.concat([data.drop('label', axis=1), one_hot_encoded], axis=1)

        for i in range(21):
            data[f'{i}x'] = data[f'{i}x'] - data['0x']
            data[f'{i}y'] = data[f'{i}y'] - data['0y']
            data[f'{i}z'] = data[f'{i}z'] - data['0z']

        data = data.sample(frac=1) 

        X = data.iloc[:, :63].values  # 获取输入数据（特征）
        y = data.iloc[:, 63:].values  # 获取输出数据（标签）


        # 将数据转换为 PyTorch 张量
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        X = normalize(X)

        return X, y
    
    if type(data) == np.ndarray:
        for i in range(21):
            data[i*3:(i+1)*3] = data[i*3:(i+1)*3] - data[0:3]

        X = torch.from_numpy(data)
        X = X.to(torch.float32)
        X = normalize(X)

        print(X.shape)

        return X, None


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
            nn.Linear(32, 2)
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(color_frame)

    imgHeight, imgWidth, _ = frame.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
            points = []
            
            for id, lm in enumerate(handLms.landmark):
                points.append(lm.x)
                points.append(lm.y)
                points.append(lm.z)
            points = np.array(points).reshape(1, -1)
            X, _ = data_processing(points)
            with torch.no_grad():
                model.eval()
                predictions = model(X)
                print(predictions)




    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)
                


    cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
    cv2.imshow('Align Example', frame)

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break


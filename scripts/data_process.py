import pandas as pd
import torch.nn as nn
import torch

import numpy as np

def data_to_points_cloud(data):

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

        X = X.reshape(-1, 21, 3)

        X = X.transpose(1, 2)
        
    
    if type(data) == np.ndarray:

        for i in range(data.shape[1]):
            if i % 3 == 0:
                data[:, i] = data[:, i] - data[:, 0]
            elif i % 3 == 1:
                data[:, i] = data[:, i] - data[:, 1]
            elif i % 3 == 2:
                data[:, i] = data[:, i] - data[:, 2]


        X = torch.from_numpy(data)
        X = X.to(torch.float32)
        #print("before normalize:", X[:, 26])
        X = normalize(X)
        #print("normalize:", X[:, 26])
        X = X.unsqueeze(0)
        X = X.reshape(-1, 21, 3)
        X = X.transpose(1, 2)

        y = None

    
    return X, y


    
def normalize(tensor):
    """
    标准化张量
    """
    # 计算每行的均值和标准差
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)

    #print("mean:", mean, "std:", std)

    # 标准化张量
    normalized_tensor = (tensor - mean) / std


    return normalized_tensor


def normalize_to_range(tensor):
    # 计算每行的最大值和最小值
    max_value, _ = torch.max(tensor, dim=1, keepdim=True)
    min_value, _ = torch.min(tensor, dim=1, keepdim=True)
    
    # 将每行的张量映射到范围为-1到1
    normalized_tensor = 2 * (tensor - min_value) / (max_value - min_value) - 1
    
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
            data[:, i*3:(i+1)*3] = data[:, i*3:(i+1)*3] - data[:, 0:3]
            
        X = torch.from_numpy(data)
        X = X.to(torch.float32)

        X = normalize(X)

        return X, None

import torch
import torch.nn as nn
import pandas as pd
import os

import numpy as np

import Data_Preprocess


# 读取训练数据
abs_data_dir = os.path.join(os.path.abspath("dataset/xiaohang_data.csv"))
xiaohang_data = pd.read_csv(abs_data_dir)
print(xiaohang_data.shape)

xiaohang_numpy = xiaohang_data.to_numpy()

abs_data_dir = "dataset/aliang_data.csv"
abs_data_dir = os.path.join(os.path.abspath("dataset/aliang_data.csv"))
aliang_data = pd.read_csv(abs_data_dir)
print(aliang_data.shape)

aliang_numpy = aliang_data.to_numpy()

train_data = np.concatenate((xiaohang_numpy, aliang_numpy), axis=0)

print(train_data.shape)

# 转换成dataframe
train_data = pd.DataFrame(train_data)

# 查看最后一列
print(train_data.iloc[:, -1].value_counts())

#随机抽样30%的数据作为测试集
test_data = train_data.sample(frac=0.2)
train_data = train_data.drop(test_data.index)

print(train_data.shape, test_data.shape)


X, y = Data_Preprocess.landmarks_to_points_cloud(train_data)
Xval, yval = Data_Preprocess.landmarks_to_points_cloud(test_data)

print(X.shape, y.shape, Xval.shape, yval.shape)
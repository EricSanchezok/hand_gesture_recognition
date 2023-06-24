import torch
import torch.nn as nn
import pandas as pd
import os

import numpy as np

import Data_Preprocess

import visualization as vis


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


X, y = Data_Preprocess.landmarks_to_linear_data(train_data)
Xval, yval = Data_Preprocess.landmarks_to_linear_data(test_data)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

X = X.to(device)
y = y.to(device)
Xval = Xval.to(device)
yval = yval.to(device)

print(X.shape, y.shape, Xval.shape, yval.shape)


class MLP(nn.Module):
    def __init__(self, in_features, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 11)
        )


    def forward(self, x):
        return self.net(x)


# 创建 MLP 模型实例
model = MLP(63, 0.1)

# 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)



num_epochs = 20
num_samples = X.shape[0]

batch_size = 32

# label_list = ['train_loss', 'val_loss']
# plotters = vis.plot_2D(vertical_num=2, label_list=label_list, figsize=(20, 14), fontsize=28)

for epoch in range(num_epochs):
    train_loss = 0.0
    for i in range(0, num_samples, batch_size):
        input = X[i:i+batch_size]
        label = y[i:i+batch_size]
 
        # 前向传播
        model.train()
        output = model(input)

        l = loss(output, label)
        
        # 反向传播和优化
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.item()

    with torch.no_grad():
        model.eval()
        y_pred = model(Xval)
        val_loss = loss(y_pred, yval).item()

    # value = []
    # value.append(train_loss)
    # value.append(val_loss)
    # plotters.draw_2Dvalue(value)

    # 打印每个 epoch 的损失
    print(f'Epoch {epoch+1}/{num_epochs}, train_Loss: {train_loss}, val_Loss: {val_loss}')
    
    if val_loss < 0.005:
        break
    




    # 保存模型
torch.save(model.state_dict(), 'model.pth')
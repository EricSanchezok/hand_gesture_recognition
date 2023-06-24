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
            nn.Linear(32, 4)
        )


    def forward(self, x):
        return self.net(x)
    

net = MLP(63, 0.1)
X = torch.rand(size=(1, 63))
for layer in net.net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)


# 创建 MLP 模型实例
model = MLP(63, 0.1)

# 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)



num_epochs = 20
num_samples = X.shape[0]


for epoch in range(num_epochs):
    train_loss = 0.0
    for i in range(0, num_samples):
        input = X[i]
        label = y[i]

        input = input.unsqueeze(0)
        label = label.unsqueeze(0)

 
        # 前向传播
        model.train()
        output = model(input)

        l = loss(output, label)
        
        # 反向传播和优化
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.item()

    train_loss /= num_samples

    model.eval()
    val_loss = loss(model(inputs_val), labels_val).item()

    # 打印每个 epoch 的损失
    print(f'Epoch {epoch+1}/{num_epochs}, train_Loss: {train_loss}, val_Loss: {val_loss}')
    
    if val_loss < 0.005:
        break
    




    # 保存模型
torch.save(model.state_dict(), 'model.pth')
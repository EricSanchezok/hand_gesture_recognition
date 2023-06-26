import torch
import torch.nn as nn
import pandas as pd
import os

import numpy as np

import Data_Preprocess

import visualization as vis
import MLP_Model


# 读取训练数据
abs_data_dir = os.path.join(os.path.abspath("dataset/data_1.csv"))
data_1 = pd.read_csv(abs_data_dir)
print(data_1.shape)

data_1_numpy = data_1.to_numpy()

abs_data_dir = "dataset/data_2.csv"
abs_data_dir = os.path.join(os.path.abspath("dataset/data_2.csv"))
data_2 = pd.read_csv(abs_data_dir)
print(data_2.shape)

data_2_numpy = data_2.to_numpy()

train_data = np.concatenate((data_1_numpy, data_2_numpy), axis=0)

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



# 创建 MLP 模型实例
model = MLP_Model.MLP(63, 0.1)

model.cuda(device=device)

# 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)



num_epochs = 20
num_samples = X.shape[0]

batch_size = 32


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

    train_loss /= num_samples

    with torch.no_grad():
        model.eval()
        y_pred = model(Xval)
        val_loss = loss(y_pred, yval).item()


    # 打印每个 epoch 的损失
    print(f'Epoch {epoch+1}/{num_epochs}, train_Loss: {train_loss}, val_Loss: {val_loss}')
    
    



# 保存模型
abs_data_dir = os.path.join(os.path.abspath("model/mlp.pth"))
torch.save(model.state_dict(), abs_data_dir)
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import hand_roscv.process_data as p_data
import hand_roscv.pointnet_model as pm
import hand_roscv.visualization as vs

import os

# 读取训练数据
file_dir = "scripts/dataset/xiaohang_data.csv"
abs_file_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), file_dir)
xiaohang_data = pd.read_csv(abs_file_dir)
print(xiaohang_data.shape)

xiaohang_numpy = xiaohang_data.to_numpy()

file_dir = "scripts/dataset/aliang_data.csv"
abs_file_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), file_dir)
aliang_data = pd.read_csv(abs_file_dir)
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


X, y = p_data.data_to_points_cloud(train_data)
Xval, yval = p_data.data_to_points_cloud(test_data)

print(X.shape, y.shape, Xval.shape, yval.shape)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

X = X.to(device)
y = y.to(device)
Xval = Xval.to(device)
yval = yval.to(device)


model = pm.get_model(num_classes=11, global_feat=True, feature_transform=False, channel=3)

model.cuda(device=device)

# 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001)


num_epochs = 20
num_samples = X.shape[0]
batch_size = 32

vs2Dvalue = vs.plot_2D(2)

for epoch in range(num_epochs):
    train_loss = 0.0
    for i in range(0, num_samples, batch_size):
        input = X[i:i+batch_size]
        label = y[i:i+batch_size]
 
        # 前向传播
        model.train()
        output, trans, trans_feat = model(input)

        l = loss(output, label)
        
        # 反向传播和优化
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.item()

    train_loss /= num_samples
    with torch.no_grad():
        model.eval()
        y_pred, _, _ = model(Xval)
        val_loss = loss(y_pred, yval).item()

    value = []
    value.append(train_loss)
    value.append(val_loss)
    vs2Dvalue.draw_2Dvalue(value)


    # 打印每个 epoch 的损失
    print(f'Epoch {epoch+1}/{num_epochs}, train_Loss: {train_loss}, val_Loss: {val_loss}')
    
    if val_loss < 0.0001:
        break
    


# 保存模型
file_dir = "scripts/model/model.pth"
abs_file_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), file_dir)
torch.save(model.state_dict(), abs_file_dir)
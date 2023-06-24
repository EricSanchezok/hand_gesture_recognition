import torch
import numpy as np
from mayavi import mlab

# 加载模型和权重
model = YourModel()  # 替换为你的模型类
model.load_state_dict(torch.load('your_model_weights.pth'))

# 获取中间层特征
input_data = torch.randn(1, 3, 64, 64)  # 替换为你的输入数据
features = model.intermediate_layers(input_data)  # 替换为你的中间层输出

# 将特征转换为numpy数组
features_np = features.detach().numpy()

# 创建3D可视化
mlab.figure(bgcolor=(1, 1, 1))  # 设置背景颜色
mlab.contour3d(features_np)
mlab.show()

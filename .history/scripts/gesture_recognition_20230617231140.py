import torch
from pointnet_model import get_model

import os


# 加载模型
model = get_model(num_classes=4, global_feat=True, feature_transform=False, channel=3)

model_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'model/model.pth')

model.load_state_dict(torch.load(model_dir))
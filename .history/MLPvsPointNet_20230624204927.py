import numpy as np
import torch
import os
import cv2

import Hand_Detector 
import Pointnet_Model
import MLP_Model



import realSense_L515_Init
import visualization as vis
import Data_Preprocess



hand_detector = Hand_Detector.mediaPipe_Hand_Detector() 

# 加载模型
Pointnet_model = Pointnet_Model.get_model(num_classes=11, global_feat=True, feature_transform=False, channel=3)
abs_model_dir = os.path.join(os.path.abspath('model/pointnet.pth'))
Pointnet_model.load_state_dict(torch.load(abs_model_dir))

# 加载模型
MLP_model = MLP_Model.MLP(63, 0.1)
print(type(MLP_model))
abs_model_dir = os.path.join(os.path.abspath('model/mlp.pth'))
MLP_model.load_state_dict(torch.load(abs_model_dir))

realSense_L515 = realSense_L515_Init.realSense()
fps = vis.FPS()

def test_model_output(model, landmarks_points, mlp=False):
    if type(model) == 'MLP_Model.MLP':
        X, _ = Data_Preprocess.landmarks_to_linear_data(landmarks_points)
    else:
        X, _ = Data_Preprocess.landmarks_to_points_cloud(landmarks_points)

    with torch.no_grad():
        model.eval()
        predictions, _, _ = model(X)
        ori_pred_val = predictions.max(1)[0]
        ori_pred_index = predictions.max(1)[1]

        pred_val = ori_pred_val.item()
        pred_index = ori_pred_index.item()

    return pred_val, pred_index

def main():

    while True:

        aligned_depth_frame, color_frame = realSense_L515.get_frame()

        if not aligned_depth_frame or not color_frame:
            continue
        
        ori_color_image = np.asanyarray(color_frame.get_data())

        show_color_image = ori_color_image.copy()

        handLandmarks_points, handworldLandmarks_points = hand_detector.get_landmarks(show_color_image, draw_fingers=True)

        mlp_value, mlp_index = test_model_output(MLP_model, handLandmarks_points)
        pointnet_value, pointnet_index = test_model_output(Pointnet_model, handworldLandmarks_points)

        print("MLP: ", mlp_index, "PointNet: ", pointnet_index)
        
        
        cv2.namedWindow('ResaultWindow', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('ResaultWindow', show_color_image)

       
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

         
if __name__ == '__main__':
  try:
    main()

  finally:
    realSense_L515.pipeline.stop()

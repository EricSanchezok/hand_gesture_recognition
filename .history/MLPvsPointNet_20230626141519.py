import numpy as np
import torch
import os
import cv2

import Hand_Detector 
import Pointnet_Model
import MLP_Model

import Process_Landmarks

import realSense_L515_Init
import visualization as vis



hand_detector = Hand_Detector.mediaPipe_Hand_Detector() 

# 加载模型
Pointnet_model = Pointnet_Model.get_model(num_classes=11, global_feat=True, feature_transform=False, channel=3)
abs_model_dir = os.path.join(os.path.abspath('model/pointnet.pth'))
Pointnet_model.load_state_dict(torch.load(abs_model_dir))

# 加载模型
MLP_model = MLP_Model.MLP(63, 0.1)
abs_model_dir = os.path.join(os.path.abspath('model/mlp.pth'))
MLP_model.load_state_dict(torch.load(abs_model_dir))

realSense_L515 = realSense_L515_Init.realSense()
fps = vis.FPS()


def main():

    while True:

        aligned_depth_frame, color_frame = realSense_L515.get_frame()

        if not aligned_depth_frame or not color_frame:
            continue
        
        ori_color_image = np.asanyarray(color_frame.get_data())

        show_color_image = ori_color_image.copy()

        handLandmarks_points, handworldLandmarks_points = hand_detector.get_landmarks(show_color_image, draw_fingers=True)

        if handworldLandmarks_points is not None:
            mlp_index = Process_Landmarks.get_pred_index(model, handworldLandmarks_points, history_size = 5, print_pred_index=False)
            pointnet_index = Process_Landmarks.get_pred_index(model, handworldLandmarks_points, history_size = 5, print_pred_index=False)

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

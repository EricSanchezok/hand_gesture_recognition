import numpy as np

import process_gesture
import process_point
import Hand_Detector 
import process_task
import Pointnet_Model

import Data_Saver

import realSense_L515_Init
import visualization as vs

import torch
import os
import cv2

save_data_mode = False

file_dir = "dataset/test_data.csv"
abs_file_dir = os.path.join(os.path.abspath(file_dir))

if save_data_mode:
    data_saver = Data_Saver.DataSaver(abs_file_dir)
    flip_hand_detector = Hand_Detector.mediaPipe_Hand_Detector()

hand_detector = Hand_Detector.mediaPipe_Hand_Detector() 

task_distributor = process_task.task_distributor()
fps = vs.FPS()

# 加载模型
model = Pointnet_Model.get_model(num_classes=11, global_feat=True, feature_transform=False, channel=3)
model_dir = os.path.join(os.path.abspath('model/model.pth'))
model.load_state_dict(torch.load(model_dir))

realSense_L515 = realSense_L515_Init.realSense()

def main():

    kalman_fliter_init = False

    while True:

        aligned_depth_frame, color_frame = realSense_L515.get_frame()

        if not aligned_depth_frame or not color_frame:
            continue
        
        ori_color_image = np.asanyarray(color_frame.get_data())

        show_color_image = ori_color_image.copy()

        handLandmarks_points, handworldLandmarks_points = hand_detector.get_landmarks(show_color_image, draw_fingers=True)

        most_index = process_gesture.get_pred_index(model, handworldLandmarks_points, print_pred_index=False)

        cv2.namedWindow('ResaultWindow', cv2.WINDOW_AUTOSIZE)

        dt = fps.showFPS(show_color_image, print_FPS=False)

        if save_data_mode:

            flip_show_color_image = cv2.flip(ori_color_image, 1)
            flip_handLandmarks_points, flip_handworldLandmarks_points = flip_hand_detector.get_landmarks(flip_show_color_image, draw_fingers=True)

            show_image = np.hstack((show_color_image, flip_show_color_image))

            show_image = cv2.resize(show_image, (0, 0), fx=0.5, fy=0.5)    

            cv2.imshow('ResaultWindow', show_image)

        else:
            kalman_fliter_3D_point, target_image, kalman_fliter_init = process_point.point_proccessing(ori_color_image, aligned_depth_frame, handLandmarks_points, realSense_L515, most_index, kalman_fliter_init, dt)
        
            task_distributor.srv_value = task_distributor.handle_all_process(most_index, kalman_fliter_3D_point)

            if target_image is not None:
                cv2.imshow('ResaultWindow', target_image)

            else:
                cv2.imshow('ResaultWindow', show_color_image)
       
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if save_data_mode:
            data_saver.readytosave(key, handworldLandmarks_points, flip=False, curPred=most_index)
            data_saver.readytosave(key, flip_handworldLandmarks_points, flip=True, curPred=most_index)


         
if __name__ == '__main__':
  try:
    main()

  finally:
    L515.pipeline.stop()

import numpy as np
import torch
import os
import cv2

import Data_Saver
import Hand_Detector 
import MLP_Model
import Process_Landmarks
import visualization as vis


save_data_mode = False

abs_data_dir = os.path.join(os.path.abspath("dataset/test_data.csv"))

if save_data_mode:
    data_saver = Data_Saver.DataSaver(abs_data_dir)
    flip_hand_detector = Hand_Detector.mediaPipe_Hand_Detector()

hand_detector = Hand_Detector.mediaPipe_Hand_Detector() 

# 加载模型
model = MLP_Model.MLP(63, 0.1)
abs_model_dir = os.path.join(os.path.abspath('model/mlp.pth'))
model.load_state_dict(torch.load(abs_model_dir))


fps = vis.FPS()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

def main():

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?).")
            continue

        ori_color_image = np.asanyarray(frame)

        show_color_image = ori_color_image.copy()

        handLandmarks_points, handworldLandmarks_points = hand_detector.get_landmarks(show_color_image, draw_fingers=True)

        current_index = Process_Landmarks.get_pred_index(model, handworldLandmarks_points, history_size = 5, print_pred_index=not save_data_mode)

        # 把current_index画到图像上
        cv2.putText(show_color_image, str(current_index), (show_color_image.shape[1] - 100, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 5)

        dt = fps.showFPS(show_color_image, print_FPS=False)

        cv2.namedWindow('ResaultWindow', cv2.WINDOW_AUTOSIZE)

        if save_data_mode:

            flip_show_color_image = cv2.flip(ori_color_image, 1)
            flip_handLandmarks_points, flip_handworldLandmarks_points = flip_hand_detector.get_landmarks(flip_show_color_image, draw_fingers=True)

            show_image = np.hstack((show_color_image, flip_show_color_image))

            show_image = cv2.resize(show_image, (0, 0), fx=0.5, fy=0.5)    

            cv2.imshow('ResaultWindow', show_image)

        else:
                cv2.imshow('ResaultWindow', show_color_image)
       
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if save_data_mode:
            data_saver.readytosave(key, handworldLandmarks_points, flip=False, curPred=current_index)
            data_saver.readytosave(key, flip_handworldLandmarks_points, flip=True, curPred=current_index)


         
if __name__ == '__main__':
  try:
    main()

  finally:
    cap.release()

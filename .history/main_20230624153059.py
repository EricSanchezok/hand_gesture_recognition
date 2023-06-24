import numpy as np

import process_gesture
import process_point
import process_hand
import process_task
import pointnet_model

import realSenseInit as rs
import visualization as vs

import rospy 
import torch
import os
import cv2

hand_detector = process_hand.mediaPipeHand() 
flip_hand_detector = process_hand.mediaPipeHand()

handLandmarks_points, results = None, None
flip_handLandmarks_points, flip_results = None, None

task_distributor = process_task.task_distributor()
fps = vs.FPS()

# 加载模型
model = pointnet_model.get_model(num_classes=11, global_feat=True, feature_transform=False, channel=3)
model_dir = os.path.join(os.path.abspath('model/model.process_taskh'))
model.load_state_dict(torch.load(model_dir))

L515 = rs.realSense()

pred_index = None

draw_fingers = True
print_pred_index = False


kalman_fliter_init = False
kalman_fliter_3D_point = None
target_image = None



def server_process_frame():

    handLandmarks_points, results = hand_detector.get_world_points(show_color_image)

    process_gesture.cal_finger_angle()

    process_gesture.get_pred()

    kalman_fliter_3D_point, target_image = process_point.point_proccessing()

    task_distributor.srv_value = task_distributor.handle_all_process(pred_index, kalman_fliter_3D_point)

def server_process_frame_savemode():
        
    handLandmarks_points, results = hand_detector.get_world_points(show_color_image, draw_finger=draw_finger)

    process_gesture.cal_finger_angle()
    process_gesture.get_pred()

    #左右翻转
    flip_show_color_image = cv2.flip(ori_color_image, 1)
    flip_handLandmarks_points, flip_result = flip_hand_detector.get_world_points(flip_show_color_image, draw_finger=draw_finger)

    # 拼接图像
    show_image = np.hstack((show_color_image, flip_show_color_image))
    # 图像尺寸减半
    show_image = cv2.resize(show_image, (0, 0), fx=0.5, fy=0.5)      
    

    cv2.imshow('ResaultWindow', show_image)


file_dir = "dataset/test_data.csv"
abs_file_dir = os.path.join(os.path.abspath(file_dir))
saveMode = False
if saveMode:
    data_saver = process_hand.DataSaver(abs_file_dir)
        
def main():

    while True:

        aligned_deprocess_taskh_frame, color_frame = L515.get_frame()

        if not aligned_deprocess_taskh_frame or not color_frame:
            continue
        
        ori_color_image = np.asanyarray(color_frame.get_data())

        show_color_image = ori_color_image.copy()

        dt = fps.showFPS(show_color_image, printFPS=False)
      

        cv2.namedWindow('ResaultWindow', cv2.WINDOW_AUTOSIZE)

        if saveMode:

            server_process_frame_savemode()

        else:

            server_process_frame()

            if target_image is not None:
                cv2.imshow('ResaultWindow', target_image)

            else:
                cv2.imshow('ResaultWindow', show_color_image)
       

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if saveMode:
            data_saver.readytosave(key, results, flip=False, curPred=pred_index)
            data_saver.readytosave(key, flip_results, flip=True, curPred=pred_index)


         
if __name__ == '__main__':
  try:
    main()

  finally:
    L515.pipeline.stop()

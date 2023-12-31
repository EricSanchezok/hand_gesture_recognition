#!/usr/bin/env python3
import numpy as np


import process_gesture as pg
import process_point as pp
import process_hand as ph
import process_task as pt
import pointnet_model as pm

import realSenseInit as rs
import visualization as vs


import rospy 
import torch
import os
import cv2



class server:
    def __init__(self) -> None:
        self.hand_detector = ph.mediaPipeHand() 
        self.flip_hand_detector = ph.mediaPipeHand()

        self.handLandmarks_points, self.results = None, None
        self.flip_handLandmarks_points, self.flip_results = None, None

        self.task_distributor = pt.task_distributor()
        self.fps = vs.FPS()

        # 加载模型
        self.model = pm.get_model(num_classes=11, global_feat=True, feature_transform=False, channel=3)
        self.model_dir = os.path.join(os.path.abspath('/model/model.pth'))
        self.model.load_state_dict(torch.load(self.model_dir))

        self.L515 = rs.realSense()

        self.pred_index = None

        self.draw_finger = True
        self.print_pred_index = False


        self.kalman_fliter_init = False
        self.kalman_fliter_3D_point = None
        self.target_image = None

    def server_get_frame(self):
        self.aligned_depth_frame, self.color_frame = self.L515.get_frame()

        if not self.aligned_depth_frame or not self.color_frame:
            return False
        
        self.ori_color_image = np.asanyarray(self.color_frame.get_data())

        self.show_color_image = self.ori_color_image.copy()

        self.dt = self.fps.showFPS(self.show_color_image, printFPS=False)

        return True

    def server_process_frame(self):

        self.handLandmarks_points, self.results = self.hand_detector.get_world_points(self.show_color_image, draw_finger=self.draw_finger)

        pg.cal_finger_angle(self)

        pg.get_pred(self)

        self.kalman_fliter_3D_point, self.target_image = pp.point_proccessing(self)

        self.task_distributor.srv_value = self.task_distributor.handle_all_process(self.pred_index, self.kalman_fliter_3D_point)

    def server_process_frame_savemode(self):
            
        self.handLandmarks_points, self.results = self.hand_detector.get_world_points(self.show_color_image, draw_finger=self.draw_finger)

        pg.cal_finger_angle(self)
        pg.get_pred(self)

        #左右翻转
        self.flip_show_color_image = cv2.flip(self.ori_color_image, 1)
        self.flip_handLandmarks_points, self.flip_result = self.flip_hand_detector.get_world_points(self.flip_show_color_image, draw_finger=self.draw_finger)

        # 拼接图像
        show_image = np.hstack((self.show_color_image, self.flip_show_color_image))
        # 图像尺寸减半
        show_image = cv2.resize(show_image, (0, 0), fx=0.5, fy=0.5)      
        
    
        cv2.imshow('ResaultWindow', show_image)


server = server()

file_dir = "/dataset/test_data.csv"
abs_file_dir = os.path.join(os.path.abspath(file_dir))
saveMode = False
if saveMode:
    data_saver = ph.DataSaver(abs_file_dir)
        
def main():


    while not rospy.is_shutdown():
      
        if server.server_get_frame():

            cv2.namedWindow('ResaultWindow', cv2.WINDOW_AUTOSIZE)

            if saveMode:
                server.server_process_frame_savemode()

            else:

                server.server_process_frame()

                if server.target_image is not None:
                    cv2.imshow('ResaultWindow', server.target_image)

                else:
                    cv2.imshow('ResaultWindow', server.show_color_image)
       

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if saveMode:
            data_saver.readytosave(key, server.results, flip=False, curPred=server.pred_index)
            data_saver.readytosave(key, server.flip_results, flip=True, curPred=server.pred_index)


         
if __name__ == '__main__':
  try:
    main()
  except rospy.ROSInterruptException:
    pass

  finally:
    server.L515.pipeline.stop()

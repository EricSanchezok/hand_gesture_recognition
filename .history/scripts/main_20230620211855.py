import numpy as np
import cv2
import os

import process_gesture as pg
import process_point as pp
import process_hand as ph
import process_task as pt

import realSenseInit as rs
import visualization as vs

    
file_dir = "dataset/aliang.csv"

abs_file_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), file_dir)
    
hand_detector = ph.mediaPipeHand()
flip_hand_detector = ph.mediaPipeHand()
data_saver = ph.DataSaver(abs_file_dir)

fps = vs.FPS()




def main():

    while True:

        aligned_depth_frame, color_frame = L515.get_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        #ori_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        ori_color_image = np.asanyarray(color_frame.get_data())

        show_color_image = ori_color_image.copy()

        points, results = hand_detector.get_world_points(show_color_image, drawPoints=True)

        dt = fps.showFPS(show_color_image, printFPS=False)

        angle_list = pg.cal_finger_angle(results, show_color_image)

        pred_index = pg.get_pred(points, angle_list, IsPrint=False)

        cv2.namedWindow('ResaultWindow', cv2.WINDOW_AUTOSIZE)

        #左右翻转
        flip_show_color_image = cv2.flip(ori_color_image, 1)
        flip_points, flip_results = flip_hand_detector.get_world_points(flip_show_color_image, drawPoints=True)

        # 拼接图像
        show_image = np.hstack((show_color_image, flip_show_color_image))
        # 图像尺寸减半
        show_image = cv2.resize(show_image, (0, 0), fx=0.5, fy=0.5)      
        
        

        cv2.imshow('ResaultWindow', show_image)



        key = cv2.waitKey(1)
        if saveMode:
            data_saver.readytosave(key, results, flip=False, curPred=pred_index)
            data_saver.readytosave(key, flip_results, flip=True, curPred=pred_index)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break






if __name__ == "__main__":

    try:
        main()
                            
    finally:
        L515.pipeline.stop()
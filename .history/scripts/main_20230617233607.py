import numpy as np
import cv2

import time

from gesture_recognition import get_pred

import point_process as pp

import os

import realSenseInit as rs
import visualization as vs
import hand_detector as hd


    
    

    
file_dir = "dataset/world_data.csv"

abs_file_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), file_dir)
    
hand_detector = hd.mediaPipeHand()

fps = vs.FPS()

data_saver = hd.DataSaver(abs_file_dir)

L515 = rs.realSense()

saveMode = False


def main():

    while True:

        aligned_depth_frame, color_frame = L515.get_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        ori_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        ori_color_image = np.asanyarray(color_frame.get_data())

        show_color_image = ori_color_image.copy()

        points, results = hand_detector.get_world_points(show_color_image, drawPoints=True)

        dt = fps.showFPS(show_color_image)

        cv2.namedWindow('ResaultWindow', cv2.WINDOW_AUTOSIZE)

        pred_val, pred_index = get_pred(points)

        fliter_point, target_image = pp.point_proccessing(pred_index, results, ori_color_image, aligned_depth_frame, L515, dt)
            
        if target_image is not None:
            cv2.imshow('ResaultWindow', target_image)
                    
        else:
            cv2.imshow('ResaultWindow', show_color_image)


        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        data_saver.readytosave(key, results, saveMode)




if __name__ == "__main__":

    try:
        main()
                            
    finally:
        L515.pipeline.stop()
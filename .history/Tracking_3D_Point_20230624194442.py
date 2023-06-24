
import pyrealsense2 as rs
import numpy as np
from filterpy.kalman import KalmanFilter

import visualization as vis


kf = KalmanFilter(dim_x=6, dim_z=3)  # 状态向量维度为3，观测向量维度为3
# 定义观测矩阵
kf.H = np.array([[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]])

q = 0.01  # 过程噪声方差
# 定义过程噪声协方差矩阵
kf.Q = np.eye(6) * q  # q为过程噪声方差

r = 0.1  # 测量噪声方差
# 定义测量噪声协方差矩阵
kf.R = np.eye(3) * r  # r为测量噪声方差

# 初始化状态向量和状态协方差矩阵
kf.P = np.eye(6)

def get_ori_depth_point(ori_color_image, aligned_depth_frame, handLandmarks_points, L515):

    img_width, img_height = ori_color_image.shape[1], ori_color_image.shape[0]

    color_image = ori_color_image.copy()
    # 设置搜索窗口大小
    search_window = 50

    if handLandmarks_points is None:
        return None, None
    
    x_list = []
    y_list = []

    for i in range(0, handLandmarks_points.shape[1], 3):
        x_list.append(handLandmarks_points[0][i])
        y_list.append(handLandmarks_points[0][i+1])

    x_list = np.array(x_list)
    y_list = np.array(y_list)

    center_x = int(np.mean(x_list))
    center_y = int(np.mean(y_list))
    
    cx, cy = min(int(x_list[8] * img_width), img_width-1), min(int(y_list[8] * img_height), img_height-1)

    target_color_image = color_image[cy-search_window:cy+search_window, cx-search_window:cx+search_window]

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image = depth_image * L515.depth_scale

    target_depth_image = depth_image[cy-search_window:cy+search_window, cx-search_window:cx+search_window]

    # 将深度大于2m的点置为0
    target_depth_image[target_depth_image > 3] = 0

    # 将深度为0的点的彩色图像置为0
    target_color_image[target_depth_image == 0] = 0

    # 如果深度图不全为0, 寻找深度图中非零点的最小值
    # 同时返回最小值的坐标
    if np.sum(target_depth_image) != 0:

        min_depth = np.min(target_depth_image[target_depth_image > 0])

        depth_pixel = [cx, cy]

        estimated_depth_point = rs.rs2_deproject_pixel_to_point(L515.camera_intrinsics, depth_pixel, min_depth)

        ori_depth_point = rs.rs2_transform_point_to_point(L515.extrinsics, estimated_depth_point)

        # 将坐标系转换为ros中的坐标系
        ori_depth_point = np.array([ori_depth_point[2], -ori_depth_point[0], -ori_depth_point[1]])

        return ori_depth_point, target_color_image

    else:
        print('您的手移动的太快了, 请缓慢移动手部')
        
    
    return None, None


# 递推加权平均滤波
def get_average_depth_point(ori_depth_point, depth_points_list):

    if ori_depth_point is None:
        return None

    depth_point_current = ori_depth_point.copy()

    depth_points_list.append(depth_point_current)
    if len(depth_points_list) > 5:
        depth_points_list.pop(0)

    depth_points = depth_points_list.copy()
    
    depth_points.sort(key=lambda x: x[0])

    depth_points = np.array(depth_points)


    average_depth_point = depth_point_current * 0.5 + depth_points.mean(axis=0) * 0.5

    return average_depth_point

def get_kalmanfliter_depth_point(average_depth_point, kalman_fliter_init, dt):

    if average_depth_point is None:
        return None, None

    if not kalman_fliter_init:
        kf.x = np.array([[average_depth_point[0]], [average_depth_point[1]], [average_depth_point[2]], [0], [0], [0]], dtype=np.float32)
        kalman_fliter_init = True
    

    #假设目标的运动模式为匀速运动，因此状态转移矩阵为：
    kf.F = np.array([[1, 0, 0, dt, 0, 0.5*dt**2],
                    [0, 1, 0, 0, dt, 0.5*dt**2],
                    [0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

    # 预测步骤
    kf.predict()

    measurement = np.array([average_depth_point[0], average_depth_point[1], average_depth_point[2]])

    # 更新步骤
    kf.update(measurement)

    fliter_depth_point = kf.x[:3, 0]

    return fliter_depth_point, kalman_fliter_init


label_list = ['ori_depth_point', 'average_depth_point', 'kalman_fliter_depth_point']
plotters = vis.plot_2D(vertical_num=3, label_list=label_list, figsize=(20, 14), fontsize=28)
depth_points_list = []

def point_proccessing(ori_color_image, aligned_depth_frame, handLandmarks_points, l515, kalman_fliter_init, dt):

    global depth_points_list

    ori_3D_point, target_image = get_ori_depth_point(ori_color_image, aligned_depth_frame, handLandmarks_points, l515)

    average_fliter_3D_point = get_average_depth_point(ori_3D_point, depth_points_list)

    kalman_fliter_3D_point, kalman_fliter_init = get_kalmanfliter_depth_point(average_fliter_3D_point, kalman_fliter_init, dt)

    value_list = []
    value_list.append(ori_3D_point[0])
    value_list.append(average_fliter_3D_point[0])
    value_list.append(kalman_fliter_3D_point[0])

    plotters.draw_2Dvalue(value_list)

    return kalman_fliter_3D_point, target_image, kalman_fliter_init
    

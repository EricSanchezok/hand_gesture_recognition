import time
import cv2
import matplotlib.pyplot as plt


class FPS:
    def __init__(self) -> None:
        self.pTime = 0
        self.cTime = 0

    def showFPS(self, img, print_FPS=False):
        self.cTime = time.time()
        fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)

        if print_FPS:
            print(fps)

        return self.cTime - self.pTime
    

class plot_2D:
    def __init__(self, vertical_num, label_list, figsize=(8, 6), fontsize=14) -> None:
        self.plt = plt
        self.plt.figure(figsize=figsize)  # 设置图表尺寸
        self.plt.rcParams.update({'font.size': fontsize})  # 设置全局字体大小
        self.horizontal_axis = []
        self.vertical_axis = [[] for _ in range(vertical_num)]
        self.count = 0
        self.plt.ion()
        self.color = ['green', 'blue', 'red', 'yellow', 'black', 'pink', 'gray', 'purple', 'orange', 'brown']

        # 检查传入的标签数量与垂直轴数量是否一致
        if len(label_list) != vertical_num:
            raise ValueError("The number of labels should be equal to the number of vertical axes.")

        self.labels = label_list
        self.linestyles = ['-','--','-.',':']  # 定义不同线条样式

    def draw_2Dvalue(self, value_list):
        self.horizontal_axis.append(self.count)
        for i in range(len(value_list)):
            self.vertical_axis[i].append(value_list[i])

        self.plt.clf()
        for i in range(len(value_list)):
            linestyle = self.linestyles[i % len(self.linestyles)]  # 使用循环的方式选择线条样式
            # 切片操作，仅选择最近的100个时刻进行显示
            x = self.horizontal_axis[-100:]
            y = self.vertical_axis[i][-100:]

            self.plt.plot(x, y, color=self.color[i], linestyle=linestyle, label=self.labels[i])


        self.plt.legend(loc='upper left')  # 将标签位置固定在右上角
        self.count += 1

        self.plt.pause(0.001)
        self.plt.ioff()

class plot_2D_XY:
    def __init__(self, label, figsize=(8, 6), fontsize=14, color='red', xlim =[-0.5, 0.5], ylim=[-0.5, 0.5], numshow=100) -> None:
        self.plt = plt
        self.plt.figure(figsize=figsize)  # 设置图表尺寸
        self.plt.rcParams.update({'font.size': fontsize})  # 设置全局字体大小
        self.horizontal_axis = []
        self.vertical_axis = []
        self.count = 0
        self.plt.ion()
        self.color = color

        self.label = label
        self.xlim = xlim
        self.ylim = ylim
        self.numshow = numshow

    def draw_2Dvalue(self, x_value, y_value):
        self.horizontal_axis.append(x_value)
        self.vertical_axis.append(y_value)

        self.plt.clf()

        # 切片操作，仅选择最近的100个时刻进行显示
        x = self.horizontal_axis[-self.numshow:]
        y = self.vertical_axis[-self.numshow:]

        # 设置坐标轴的值的范围
        self.plt.xlim(self.xlim[0], self.xlim[1])
        self.plt.ylim(self.ylim[0], self.ylim[1])


        # 绘制散点图
        self.plt.scatter(x, y, color=self.color, label=self.label)

        self.plt.legend(loc='upper left')  # 将标签位置固定在右上角
        self.count += 1

        self.plt.pause(0.001)
        self.plt.ioff()

import numpy as np

class plot_3D:
    def __init__(self, title, figsize=(8, 6), fontsize=14, color='red', xlim =[0.0, 1.0], ylim=[-0.5, 0.5], zlim=[-0.5, 0.5], numshow=100) -> None:
        self.plt = plt
        self.fig = self.plt.figure(figsize=figsize)  # 设置图表尺寸
        self.plt.rcParams.update({'font.size': fontsize})  # 设置全局字体大小

        self.ax = self.plt.axes(projection="3d")
        self.ax.view_init(30, 0)

        self.x_list = []
        self.y_list = []
        self.z_list = []

        self.count = 0

        self.color = color
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.numshow = numshow

        # 在限制范围中生成一个随机点
        self.x_t = np.random.uniform(self.xlim[0], self.xlim[1])
        self.y_t = np.random.uniform(self.ylim[0], self.ylim[1])
        self.z_t = np.random.uniform(self.zlim[0], self.zlim[1])


    def draw_3Dvalue(self, x_value, y_value, z_value):
        self.x_list.append(x_value)
        self.y_list.append(y_value)
        self.z_list.append(z_value)

        self.ax.clear()

        # 切片操作，仅选择最近的100个时刻进行显示
        x = self.x_list[-self.numshow:]
        y = self.y_list[-self.numshow:]
        z = self.z_list[-self.numshow:]

        self.ax.set_xlim3d(self.xlim[0], self.xlim[1])
        self.ax.set_ylim3d(self.ylim[0], self.ylim[1])
        self.ax.set_zlim3d(self.zlim[0], self.zlim[1])

        self.ax.set_zlabel('Z', fontdict={'color': 'red'})
        self.ax.set_ylabel('Y', fontdict={'color': 'red'})
        self.ax.set_xlabel('X', fontdict={'color': 'red'})

        # 绘制散点图        
        self.ax.scatter3D(0, 0, 0, color='green', s=40)
        self.ax.text(0, 0, 0, 'camera', color='green')

        self.ax.scatter3D(x, y, z, color=self.color, s=40)
        self.ax.scatter3D(self.x_t, self.y_t, self.z_t, color='blue', s=40)
        self.ax.text(self.x_t, self.y_t, self.z_t, 'target', color='blue')
        self.plt.title(self.title)

        if self.cal_dis(x_value, y_value, z_value) < 0.1:
            self.x_t = np.random.uniform(self.xlim[0], self.xlim[1])
            self.y_t = np.random.uniform(self.ylim[0], self.ylim[1])
            self.z_t = np.random.uniform(self.zlim[0], self.zlim[1])

        self.count += 1

        self.plt.pause(0.001)

    def cal_dis(self, x, y, z):
        return np.sqrt((x - self.x_t) ** 2 + (y - self.y_t) ** 2 + (z - self.z_t) ** 2)


    


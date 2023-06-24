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
    
import matplotlib.pyplot as plt

class plot_2D:
    def __init__(self, vertical_num, label_list, figsize=(8, 6), fontsize=14) -> None:
        self.plt = plt
        self.plt.figure(figsize=figsize)  # 设置图表尺寸
        self.plt.rcParams.update({'font.size': fontsize})  # 设置全局字体大小
        self.horizontal_axis = []
        self.vertical_axis = [[] for _ in range(vertical_num)]
        self.count = 0
        self.plt.ion()
        self.color = ['red', 'blue', 'green', 'yellow', 'black', 'pink', 'gray', 'purple', 'orange', 'brown']

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
            self.plt.plot(self.horizontal_axis, self.vertical_axis[i], color=self.color[i], label=self.labels[i])

        self.plt.legend()  # 显示图例

        self.count += 1

        self.plt.pause(0.001)
        self.plt.ioff()

    
# class plot_2D:
#     def __init__(self, vertical_num, figsize=(8, 6)) -> None:
#         self.plt = plt
#         self.plt.figure(figsize=figsize)  # 设置图表尺寸
#         self.horizontal_axis = []
#         self.vertical_axis = [[] for _ in range(vertical_num)]
#         self.count = 0
#         self.plt.ion()
#         self.color = ['red', 'blue', 'green', 'yellow', 'black', 'pink', 'gray', 'purple', 'orange', 'brown']

#     def draw_2Dvalue(self, value_list):
#         self.horizontal_axis.append(self.count)
#         for i in range(len(value_list)):
#             self.vertical_axis[i].append(value_list[i])

#         self.plt.clf()
#         for i in range(len(value_list)):
#             self.plt.plot(self.horizontal_axis, self.vertical_axis[i], color=self.color[i])

#         self.count += 1

#         self.plt.pause(0.001)
#         self.plt.ioff()
    


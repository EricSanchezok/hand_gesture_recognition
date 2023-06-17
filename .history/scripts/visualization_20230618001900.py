import time
import cv2
import matplotlib.pyplot as plt


class FPS:
    def __init__(self) -> None:
        self.pTime = 0
        self.cTime = 0

    def showFPS(self, img, printFPS=False):
        self.cTime = time.time()
        fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)

        if printFPS:
            print(fps)

        return self.cTime - self.pTime
    


class plot_2D:
    def __init__(self, vertical_num) -> None:

        self.plt = plt
        self.horizontal_axis = []
        # 创建vertical_num个列表
        self.vertical_axis = [[] for i in range(vertical_num)]
        self.count = 0
        self.plt.ion()

        self.color = ['red', 'blue', 'green', 'yellow', 'black', 'pink', 'gray', 'purple', 'orange', 'brown']

    def draw_2Dvalue(self, value_list):
         
        self.horizontal_axis.append(self.count)
        for i in range(len(value_list)):
            self.vertical_axis[i].append(value_list[i])

        self.plt.clf()
        for i in range(len(value_list)):
            self.plt.plot(self.horizontal_axis, self.vertical_axis[i], color=self.color[i])

        self.count += 1

        self.plt.pause(0.001)
        self.plt.ioff()

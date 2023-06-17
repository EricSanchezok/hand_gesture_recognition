import time
import cv2


class FPS:
    def __init__(self) -> None:
        self.pTime = 0
        self.cTime = 0

    def showFPS(self, img):
        self.cTime = time.time()
        fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, str(int(fps)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 5)

        return self.cTime - self.pTime
    

ai = []
ori_depths = []
average_depths = []
fliter_depths = []
count = 0
plt.ion()

def draw_2Dvalue(value)

        # ori_depths.append(ori_depth_point[0])
        # average_depths.append(average_depth_point[0])
        # fliter_depths.append(fliter_depth_point[0])
        # ai.append(count)
        # plt.clf()             
        # plt.plot(ai, ori_depths, color='red')
        # plt.plot(ai, average_depths, color='blue')
        # plt.plot(ai, fliter_depths, color='green')
        # plt.pause(0.001)      
        # plt.ioff()           
        # print("a:", ori_depth_point[0], "b:", average_depth_point[0], "c:", fliter_depth_point[0])
        # count += 1
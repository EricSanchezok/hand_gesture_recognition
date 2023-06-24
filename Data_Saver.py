
class DataSaver:
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.file = open(self.file_name, 'a')
        self.save_num = 0
        self.save_label = 0
        self.start_save = False


    def _writeData(self, handworldLandmarks_points, label:int):
        if handworldLandmarks_points is not None:
            for i in range(0, handworldLandmarks_points.shape[1]):
                self.file.write(str(handworldLandmarks_points[0][i]) + ",")
            self.file.write(label.__str__() + "\n")


    def readytosave(self, key, handworldLandmarks_points, flip=False, curPred=0):

        if key & 0xFF == ord('k'):
            # 取反
            if flip == False:
                self.start_save = not self.start_save
                self.save_num = 0

        #0状态
        if key & 0xFF == ord('0'):
            self.save_label = 0    
        #1状态
        if key & 0xFF == ord('1'):
            self.save_label = 1
        #2状态
        if key & 0xFF == ord('2'):
            self.save_label = 2
        #3状态
        if key & 0xFF == ord('3'):
            self.save_label = 3
        #4状态
        if key & 0xFF == ord('4'):
            self.save_label = 4
        #5状态
        if key & 0xFF == ord('5'):
            self.save_label = 5
        #6状态
        if key & 0xFF == ord('6'):
            self.save_label = 6
        #7状态
        if key & 0xFF == ord('7'):
            self.save_label = 7
        #8状态
        if key & 0xFF == ord('8'):
            self.save_label = 8
        #9状态
        if key & 0xFF == ord('9'):
            self.save_label = 9
        #ok状态
        if key & 0xFF == ord('a'):
            self.save_label = 10

        print("状态:", self.start_save, "当前预测:", curPred, "编号:", self.save_label, "次数:", self.save_num)
        if self.start_save:
            self._writeData(handworldLandmarks_points, self.save_label)
            self.save_num += 1

     
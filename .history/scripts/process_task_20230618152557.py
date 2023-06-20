



class task_distributor:
    def __init__(self) -> None:
        self.history_index_list = []

        self.stable_index_list = []

        self.current_index = None

        self.action_signal = None

    def task_preprocessing(self, pred_index):
            
        # 当一个index连续出现3次时，被认为是有效的任务
        self.history_index_list.append(pred_index)
        if len(self.history_index_list) > 10:
            self.history_index_list.pop(0)

        # 选取list中出现次数最多的元素
        most_index = max(self.history_index_list, key=self.history_index_list.count)

        # most_index的数量
        most_index_count = self.history_index_list.count(most_index)

        if most_index_count >= 3:
            self.stable_index_list.append(most_index)
            if len(self.stable_index_list) > 5:
                self.stable_index_list.pop(0)
            self.current_index = most_index
                

    def get_action_signal(self):

        # 动作顺序为5,0,5时，被认为触发了"切换任务"的动作
        if len(self.stable_index_list) == 5:
            if self.stable_index_list[4] == 5 and self.stable_index_list[3] == 0 and self.stable_index_list[2] == 5:
                self.action_signal = 1
            else:
                self.action_signal = 0


        print(self.stable_index_list)


    










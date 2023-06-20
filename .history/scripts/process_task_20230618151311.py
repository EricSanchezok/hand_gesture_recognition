



class task_distributor:
    def __init__(self) -> None:
        self.history_index_list = []
        self.last_index = None
        self.current_index = None

        self.action_signal = None

    def task_preprocessing(self, pred_index):
            
        # 当一个index连续出现5次时，被认为是有效的任务
        self.history_index_list.append(pred_index)
        if len(self.history_index_list) > 10:
            self.history_index_list.pop(0)

        # 选取list中出现次数最多的元素
        most_index = max(self.history_index_list, key=self.history_index_list.count)

        # most_index的数量
        most_index_count = self.history_index_list.count(most_index)

        if most_index_count >= 3:
            self.last_index = self.current_index
            self.current_index = most_index

    def task_assignment(self, last_index, current_index):

        # 上一个动作时0，当前动作是5，认为进行了一次“张开”
        if last_index == 0 and current_index == 5:













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

    def get_action_signal(self):

        # 上一个动作时0，当前动作是5，认为触发了一次"张开手掌"动作信号
        if self.last_index == 0 and self.current_index == 5:
            self.action_signal = "open_hand"

        # 上一个动作时5，当前动作是0，认为触发了一次"握拳"动作信号
        elif self.last_index == 5 and self.current_index == 0:
            self.action_signal = "close_hand"


        print(self.action_signal)


    














class task_distributor:
    def __init__(self) -> None:
        self.history_index_list = []

        self.stable_index_list = []

        self.current_index = None

        self.action_signal = None

        self.work_mode = 'position'

        self.init_value = None
        self.if_init_value = False


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
            if most_index != self.current_index:
                self.stable_index_list.append(most_index)
                if len(self.stable_index_list) > 3:
                    self.stable_index_list.pop(0)
                self.current_index = most_index
                

    def get_action_signal(self):

        self.action_signal = 0

        # 动作顺序为5,0,5时，被认为触发了"切换任务"的动作
        if len(self.stable_index_list) == 3:
            if self.stable_index_list[2] == 5 and self.stable_index_list[1] == 0 and self.stable_index_list[0] == 5:
                self.action_signal = 1


    def process_action_signal(self, pred_index):

        self.task_preprocessing(pred_index)

        self.get_action_signal()

        if self.action_signal == 1:
            if self.work_mode == 'position':
                self.work_mode = 'angle'
            elif self.work_mode == 'angle':
                self.work_mode = 'position'
            self.stable_index_list.pop(0)
            self.stable_index_list.pop(0)

            self.if_init_value = False

    def process_current_index(self, fliter_point):
        if fliter_point is None:
            return None
        
        if self.current_index == 1 and self.work_mode == 'position':
            if self.if_init_value == False:
                self.init_value = fliter_point
                self.if_init_value = True

            position_diff = fliter_point - self.init_value
            print("位置跟随: ", position_diff)

            return position_diff

        if self.current_index == 1 and self.work_mode == 'angle':

            if self.if_init_value == False:
                self.init_value = fliter_point
                self.if_init_value = True

            position_diff = fliter_point - self.init_value
            print("角度跟随: ", position_diff)

            return position_diff

        




        


    










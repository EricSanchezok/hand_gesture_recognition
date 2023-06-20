


history_index_list = []

last_index = None
current_index = None


def task_assignment(pred_index):

    global history_index_list, last_index, current_index

    # 当一个index连续出现5次时，被认为是有效的任务
    history_index_list.append(pred_index)
    if len(history_index_list) > 10:
        history_index_list.pop(0)

    # 选取list中出现次数最多的元素
    most_index = max(history_index_list, key=history_index_list.count)

    # most_index的数量
    most_index_count = history_index_list.count(most_index)

    if most_index_count >= 3:
        last_index = current_index
        current_index = most_index


    
    #print('last_index: ', last_index, 'current_index: ', current_index)
        








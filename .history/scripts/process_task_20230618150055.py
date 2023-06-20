


history_task_list = []

def task_assignment(pred_index):

    # 当一个index连续出现5次时，认为是一个任务
    history_task_list.append(pred_index)
    if len(history_task_list) > 10:
        history_task_list.pop(0)

    # 选取list中出现次数最多的元素


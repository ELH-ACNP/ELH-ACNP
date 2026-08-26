

class Satellite(object):
    execution_list: set
    satellite_id: int

    def __init__(self, bandwidth, id, longitude_deg=None):
        self.execution_list = set()
        self.satellite_id = id
        self.bandwidth = bandwidth
        self.longitude_deg = longitude_deg

    def add_task(self, task):
        """
        添加一个任务
        :param task: 待添加的任务
        :return: None
        """
        self.execution_list.add(task)

    def update_execution_list(self, deleted_task_set):
        """
        删除一个任务组
        :param deleted_task_set:待删除的任务集合
        :return: None
        """
        self.execution_list.difference_update(deleted_task_set)

    def clear_execution_list(self):
        """
        清空任务组
        :return: None
        """
        self.execution_list.clear()






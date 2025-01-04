from __future__ import annotations
import numpy as np
from collections import defaultdict

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite

class PredictiveDispatcher(BaseDispatcher):
    def __init__(self, queue_weight: float = 0.6, distance_weight: float = 0.4, history_window: int = 100):
        super().__init__()
        self.name = "PredictiveDispatcher"
        self.queue_weight = queue_weight
        self.distance_weight = distance_weight
        self.history_window = history_window
        
        # 历史等待时间记录
        self.load_site_history = defaultdict(list)
        self.dump_site_history = defaultdict(list)
        self.np_random = np.random.RandomState()

    def update_wait_time(self, site_name: str, wait_time: float, is_load_site: bool = True):
        """更新站点的历史等待时间"""
        history = self.load_site_history if is_load_site else self.dump_site_history
        history[site_name].append(wait_time)
        if len(history[site_name]) > self.history_window:
            history[site_name].pop(0)

    def predict_wait_time(self, site_name: str, is_load_site: bool = True) -> float:
        """预测站点的等待时间"""
        history = self.load_site_history if is_load_site else self.dump_site_history
        if not history[site_name]:
            return 0.0
        recent_history = history[site_name][-10:]
        weights = np.linspace(0.5, 1.0, len(recent_history))
        return np.average(recent_history, weights=weights)

    def calculate_score(self, site, current_position, is_load_site: bool) -> float:
        """计算站点得分"""
        # 计算队列得分
        queue_length = site.parking_lot.queue_status["total"][int(site.env.now)]
        coming_trucks = getattr(site, 'coming_truck_num', 0)
        total_queue = queue_length + coming_trucks
        queue_score = 1.0 / (1.0 + total_queue)
        
        # 计算预测等待时间得分
        predicted_wait = self.predict_wait_time(site.name, is_load_site)
        wait_score = 1.0 / (1.0 + predicted_wait)
        
        # 计算距离得分
        distance = np.linalg.norm(np.array(current_position.position) - np.array(site.position))
        distance_score = 1.0 / (1.0 + distance)
        
        # 综合得分
        return (self.queue_weight * (0.7 * queue_score + 0.3 * wait_score) + 
                self.distance_weight * distance_score)

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """初始分配装载点"""
        # 获取其他卡车信息并统计目标装载点
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, LoadSite)]
        
        # 更新装载点的即将到达卡车数
        for load_site in mine.load_sites:
            load_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == load_site.name)
        
        # 计算所有装载点得分
        site_scores = [self.calculate_score(site, truck.current_location, True) 
                      for site in mine.load_sites]
        
        return int(np.argmax(site_scores))

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """选择卸载点"""
        # 获取其他卡车信息并统计目标卸载点
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, DumpSite)]
        
        # 更新卸载点的即将到达卡车数
        for dump_site in mine.dump_sites:
            dump_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == dump_site.name)
        
        # 计算所有卸载点得分
        site_scores = [self.calculate_score(site, truck.current_location, False) 
                      for site in mine.dump_sites]
        
        return int(np.argmax(site_scores))

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """返回装载点"""
        # 获取其他卡车信息并统计目标装载点
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, LoadSite)]
        
        # 更新装载点的即将到达卡车数
        for load_site in mine.load_sites:
            load_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == load_site.name)
        
        # 计算所有装载点得分
        site_scores = [self.calculate_score(site, truck.current_location, True) 
                      for site in mine.load_sites]
        
        return int(np.argmax(site_scores))


if __name__ == "__main__":
    dispatcher = PredictiveDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))
    print(dispatcher.total_order_count, dispatcher.init_order_count)
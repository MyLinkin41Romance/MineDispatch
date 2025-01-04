from __future__ import annotations
import numpy as np

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite
from openmines.src.charging_site import ChargingSite

class PriorityDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "PriorityDispatcher"
        # 权重配置
        self.queue_weight = 0.6    # 队列权重更大，优先考虑避免拥堵 
        self.distance_weight = 0.4  # 距离其次，在队列相近时选择较近的
        self.np_random = np.random.RandomState()

    def calculate_queue_score(self, site) -> float:
        """队列得分：队列越短分数越高"""
        queue_length = site.parking_lot.queue_status["total"][int(site.env.now)]
        coming_trucks = getattr(site, 'coming_truck_num', 0)
        total_queue = queue_length + coming_trucks
        return 1.0 / (1.0 + total_queue)

    def calculate_distance_score(self, from_position, to_position) -> float:
        """距离得分：距离越近分数越高"""
        distance = np.linalg.norm(np.array(from_position.position) - np.array(to_position.position))
        return 1.0 / (1.0 + distance)

    def calculate_priority(self, truck: "Truck", site) -> float:
        """计算综合优先级分数"""
        queue_score = self.calculate_queue_score(site)
        distance_score = self.calculate_distance_score(truck.current_location, site)
        
        return (self.queue_weight * queue_score + 
                self.distance_weight * distance_score)

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        """初始分配装载点"""
        # 验证卡车位置
        assert isinstance(truck.current_location, ChargingSite), "Truck must be at charging site for initial dispatch"
        
        # 获取其他卡车信息并统计目标装载点
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, LoadSite)]
        
        # 更新装载点的即将到达卡车数
        for load_site in mine.load_sites:
            load_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == load_site.name)
        
        # 计算所有装载点的优先级
        site_priorities = []
        for load_site in mine.load_sites:
            priority = self.calculate_priority(truck, load_site)
            site_priorities.append((load_site, priority))
        
        # 选择优先级最高的装载点
        best_site = max(site_priorities, key=lambda x: x[1])[0]
        return mine.load_sites.index(best_site)

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        """选择卸载点"""
        # 验证卡车状态
        assert isinstance(truck.current_location, LoadSite), "Truck must be at load site for haul dispatch"
        assert truck.truck_load > 0, "Truck must be loaded for haul dispatch"
        
        # 获取其他卡车信息并统计目标卸载点
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, DumpSite)]
        
        # 更新卸载点的即将到达卡车数
        for dump_site in mine.dump_sites:
            dump_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == dump_site.name)
        
        # 计算所有卸载点的优先级
        site_priorities = []
        for dump_site in mine.dump_sites:
            priority = self.calculate_priority(truck, dump_site)
            site_priorities.append((dump_site, priority))
        
        # 选择优先级最高的卸载点
        best_site = max(site_priorities, key=lambda x: x[1])[0]
        return mine.dump_sites.index(best_site)

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        """返回装载点"""
        # 验证卡车状态
        assert isinstance(truck.current_location, DumpSite), "Truck must be at dump site for back dispatch"
        assert truck.truck_load == 0, "Truck must be empty for back dispatch"
        
        # 获取其他卡车信息并统计目标装载点
        other_trucks = [t for t in mine.trucks if t.name != truck.name]
        target_locations = [t.target_location for t in other_trucks 
                          if t.target_location is not None and isinstance(t.target_location, LoadSite)]
        
        # 更新装载点的即将到达卡车数
        for load_site in mine.load_sites:
            load_site.coming_truck_num = sum(1 for loc in target_locations if loc.name == load_site.name)
        
        # 计算所有装载点的优先级
        site_priorities = []
        for load_site in mine.load_sites:
            priority = self.calculate_priority(truck, load_site)
            site_priorities.append((load_site, priority))
        
        # 选择优先级最高的装载点
        best_site = max(site_priorities, key=lambda x: x[1])[0]
        return mine.load_sites.index(best_site)


if __name__ == "__main__":
    dispatcher = PriorityDispatcher()
    print(dispatcher.give_init_order(1,2))
    print(dispatcher.give_haul_order(1,2))
    print(dispatcher.give_back_order(1,2))
    print(dispatcher.total_order_count, dispatcher.init_order_count)
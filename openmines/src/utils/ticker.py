import json
import logging
import math
import time, os
from pathlib import Path
import numpy as np

from openmines.src.utils.event import EventPool
from openmines.src.utils.logger import MineLogger
from functools import reduce

# 项目主路径
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent
LOG_FILE_PATH = PROJECT_ROOT_PATH / "src" / "data" / "logs"

# 车辆状态
IDLE = 1  # 在充电站
MOVING = 2  # 从充电站到第一个装载点, 从卸载区到装载区, 从装载区到卸载区
WAITING_FOR_SHOVEL = 3  # 等待装载
LOADING = 4  # 装载中
WAITING_FOR_DUMPER = 7  # 等待卸载
UNLOADING = 8  # 卸载中
REPAIRING = 10  # 维修中
UNREPAIRABLE = 11  # 无法维修，返回充电场

class TickGenerator:
    """
    读取矿山下每个Truck的EventPool进行分析统计
    每一分钟相当于一个tick
    """

    def __init__(self, mine:"Mine", tick_num=60*8):
        self.mine = mine
        self.tick_num = tick_num
        self.ticks = dict()
        self.result_path = os.getcwd() + '/results'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)


    def run(self):
        """
        获取矿山下每个Truck的EventPool并进行总体分析
        1.获取需要dump的对象
        2.在每个tick中按照对象类别分别进行统计，形成字典格式
        :return:
        """
        # 读取mine的配置文件 获取 装载点(铲车、停车场)、卸载点（卸载位、停车场）、充电区 在图像中的位置
        load_sites = self.mine.load_sites
        dump_sites = self.mine.dump_sites
        charging_site = self.mine.charging_site
        trucks = self.mine.trucks

        # 遍历产生每个tick
        for tick in range(self.tick_num):
            cur_time = tick
            truck_states = dict()
            load_site_states = dict()
            dump_site_states = dict()
            shovel_states = dict()
            dumper_states = dict()

            for truck in self.mine.trucks:
                event_pool = truck.event_pool
                past_events = event_pool.get_event_by_time(cur_time+0.1)
                future_events = event_pool.get_event_by_time(cur_time+0.1, mode="future")
                status = None
                position = None
                truck_position = np.array([0.0, 0.0])

                if len(past_events) == 0:
                    status = IDLE  # 车辆状态为 IDLE (1)，表示在充电站
                    truck_position = np.array(charging_site.position) + np.array([0.0, 0.0])

                if len(past_events) == 2 and past_events[-1].event_type == "init":
                    status = MOVING  # 车辆状态为 MOVING (2)，表示正在移动
                    last_event = past_events[-1]
                    cur_time_delta = cur_time - last_event.info["start_time"]
                    time_ratio = cur_time_delta / (last_event.info["est_end_time"] - last_event.info["start_time"])
                    target_load_site = self.mine.get_dest_obj_by_name(last_event.info["target_location"])
                    direction = np.array(target_load_site.position) - np.array(charging_site.position)
                    truck_position = np.array(charging_site.position) + time_ratio * direction

                if past_events[-1].event_type == "wait shovel":
                    status = WAITING_FOR_SHOVEL  # 车辆状态为 WAITING_FOR_SHOVEL (3)，表示等待铲车装载
                    truck_cur_location_name = truck.current_location.name
                    cur_load_site = self.mine.get_dest_obj_by_name(truck_cur_location_name)
                    parkinglot = cur_load_site.parking_lot
                    queue_index = past_events[-1].info["queue_index"]
                    shovel_name = past_events[-1].info["shovel"]
                    shovel = self.mine.get_service_vehicle_by_name(shovel_name)
                    truck_position = np.array(shovel.position) - np.array([0.05, 0.0]) - (queue_index+1)*np.array([0.035,0.0])

                if past_events[-1].event_type == "get shovel":
                    status = LOADING  # 车辆状态为 LOADING (4)，表示正在装载
                    shovel = self.mine.get_service_vehicle_by_name(past_events[-1].info["shovel"])
                    loading_position = np.array(shovel.position) - np.array([0.05, 0.0])
                    truck_position = loading_position

                if past_events[-1].event_type == "haul":
                    status = MOVING  # 车辆状态为 MOVING (2)，表示正在运输
                    last_event = past_events[-1]
                    cur_time_delta = cur_time - last_event.info["start_time"]
                    end_time = last_event.info["end_time"] if last_event.info["end_time"] is not None else last_event.info["est_end_time"]
                    time_ratio = cur_time_delta / (end_time - last_event.info["start_time"])
                    cur_load_site = self.mine.get_dest_obj_by_name(last_event.info["start_location"])
                    target_unload_site = self.mine.get_dest_obj_by_name(last_event.info["target_location"])
                    direction = np.array(target_unload_site.position) - np.array(cur_load_site.position)
                    truck_position = np.array(cur_load_site.position) + time_ratio * direction

                if past_events[-1].event_type == "wait dumper":
                    status = WAITING_FOR_DUMPER  # 车辆状态为 WAITING_FOR_DUMPER (7)，表示等待卸载车
                    truck_cur_location_name = truck.current_location.name
                    parkinglot = self.mine.get_dest_obj_by_name(truck_cur_location_name).parking_lot
                    truck_position = np.array(parkinglot.position)

                if past_events[-1].event_type == "get dumper":
                    status = UNLOADING  # 车辆状态为 UNLOADING (8)，表示正在卸载
                    dumper = self.mine.get_service_vehicle_by_name(past_events[-1].info["dumper"])
                    truck_position = np.array(dumper.position)

                if past_events[-1].event_type == "unhaul":
                    status = MOVING  # 车辆状态为 MOVING (2)，表示正在返回
                    last_event = past_events[-1]
                    cur_time_delta = cur_time - last_event.info["start_time"]
                    end_time = last_event.info["end_time"] if last_event.info["end_time"] is not None else last_event.info["est_end_time"]
                    time_ratio = cur_time_delta / (end_time - last_event.info["start_time"])
                    cur_unload_site = self.mine.get_dest_obj_by_name(last_event.info["start_location"])
                    target_load_site = self.mine.get_dest_obj_by_name(last_event.info["target_location"])
                    direction = np.array(target_load_site.position) - np.array(cur_unload_site.position)
                    truck_position = np.array(cur_unload_site.position) + time_ratio * direction

                if "breakdown" in past_events[-1].event_type:
                    status = REPAIRING  # 车辆状态为 REPAIRING (10)，表示维修中
                    truck_position = truck_position

                if "unrepairable" in past_events[-1].event_type:
                    status = UNREPAIRABLE  # 车辆状态为 UNREPAIRABLE (11)，表示无法维修，返回充电站
                    truck_position = np.array(self.mine.charging_site.position).astype(float)

                truck_state = {
                    "name": truck.name,
                    "time": cur_time,
                    "state": status,
                    "position": list(truck_position if truck_position is not None else (0.5, 0.5)),
                    "dest_code": truck.target_location.name if truck.target_location else "Unknown"
                }
                truck_states[truck.name] = truck_state
                
            # 统计每一个loadsite的信息
            for loadsite in self.mine.load_sites:
                # 统计停车场的信息
                ## 获得停车场的排队数量
                total_queue_length = loadsite.parking_lot.queue_status["total"].get(cur_time,0)
                ## 统计loadsite的信息
                loadsite_status = loadsite.status.get(cur_time,{"produced_tons":0,"service_count":0})
                load_site_tick = {
                    "name":loadsite.name,
                    "tons":loadsite_status["produced_tons"],
                    "service_count":loadsite_status["service_count"],
                    "time":cur_time,
                    "total_queue_length":total_queue_length,
                    "shovel_queue_length":{},
                    "position":list(loadsite.position)
                }

                # 统计shovel的信息
                for shovel in loadsite.shovel_list:
                    # 获得shovel的排队数量
                    shovel_queue_length = loadsite.parking_lot.queue_status[shovel.name].get(cur_time,0)
                    load_site_tick["shovel_queue_length"][shovel.name] = shovel_queue_length
                    # 统计shovel的信息
                    shovel_cur_state = shovel.status.get(cur_time,{"produced_tons":0,"service_count":0})
                    shovel_tick = {
                        "name":shovel.name,
                        "tons":shovel_cur_state["produced_tons"],
                        "service_count":shovel_cur_state["service_count"],
                        "time":cur_time,
                        "queue_length":shovel_queue_length,
                        "position":(shovel.position),
                    }
                    shovel_states[shovel.name] = shovel_tick
                load_site_states[loadsite.name] = load_site_tick

            # 统计每一个dumpsite的信息
            for dumpsite in self.mine.dump_sites:
                # 统计停车场的信息
                ## 获得停车场的排队数量
                total_queue_length = dumpsite.parking_lot.queue_status["total"].get(cur_time,0)
                # 统计dumpsite的信息
                dump_site_status = dumpsite.status.get(cur_time,{"produced_tons":0,"service_count":0})
                dump_site_tick = {
                    "name":dumpsite.name,
                    "tons":dump_site_status["produced_tons"],
                    "service_count":dump_site_status["service_count"],
                    "time":cur_time,
                    "total_queue_length":total_queue_length,
                    "dumper_queue_length":{},
                    "position":list(dumpsite.position)
                }

                # 统计dumper的信息
                for dumper in dumpsite.dumper_list:
                    # 获得dumper的排队数量
                    dumper_queue_length = dumpsite.parking_lot.queue_status[dumper.name].get(cur_time,0)
                    dump_site_tick["dumper_queue_length"][dumper.name] = dumper_queue_length
                    # 统计dumper的信息
                    dumper_cur_state = dumper.status.get(cur_time,{"produced_tons":0,"service_count":0})

                    dumper_tick = {
                        "name":dumper.name,
                        "tons":dumper_cur_state["produced_tons"],
                        "service_count":dumper_cur_state["service_count"],
                        "time":cur_time,
                        "queue_length":dumper_queue_length,
                        "position":list(dumper.position),
                    }
                    dumper_states[dumper.name] = dumper_tick
                dump_site_states[dumpsite.name] = dump_site_tick

            # 统计矿山整体信息
            mine_status = self.mine.status.get(cur_time,{"produced_tons":0,"service_count":0,"working_truck_count":0,"waiting_truck_count":0,"load_unload_truck_count":0,"moving_truck_count":0,"repairing_truck_count":0,"road_jam_count":0,"road_repair_count":0,"truck_repair":0,"truck_unrepairable":0,"random_event_count":0})
            mine_tick = {
                "time":cur_time,
                # KPIs
                "produced_tons":mine_status["produced_tons"],
                "service_count":mine_status["service_count"],
                # stats
                "working_truck_count": mine_status["working_truck_count"],
                "waiting_truck_count": mine_status["waiting_truck_count"],
                "load_unload_truck_count": mine_status["load_unload_truck_count"],
                "moving_truck_count": mine_status["moving_truck_count"],
                "repairing_truck_count": mine_status["repairing_truck_count"],
                # event stats
                "road_jam_count": mine_status["road_jam_count"],
                "road_repair_count": mine_status["road_repair_count"],
                "truck_repair": mine_status["truck_repair"],
                "truck_unrepairable": mine_status["truck_unrepairable"],
                "random_event_count":mine_status["random_event_count"]
            }


            tick_packet = {
                "tick":cur_time,
                "truck_states":truck_states,
                "load_site_states":load_site_states,
                "dump_site_states":dump_site_states,
                "shovel_states":shovel_states,
                "dumper_states":dumper_states,
                "mine_states":mine_tick
            }
            self.ticks[cur_time] = tick_packet

        # 添加一些分析性非实时性的统计数据
        # 统计卸载区产量
        print(f"{self.mine.name} summary: {self.ticks[cur_time]['mine_states']['produced_tons']} tons")
        # 统计MatchingFactor
        # paper:Match factor for heterogeneous truck and loader fleets
        shovels = [shovel for load_site in self.mine.load_sites for shovel in load_site.shovel_list]
        trucks = self.mine.trucks
        truck_cycle_time_avg = np.mean([truck.truck_cycle_time for truck in trucks])

        num_trucks = len(self.mine.trucks)
        num_shovels = len(shovels)
        loading_time:np.array = np.zeros((num_trucks,num_shovels))
        for i,truck in enumerate(trucks):
            for j,shovel in enumerate(shovels):
                loading_time[i, j] = round((truck.truck_capacity / shovel.shovel_tons), 1) * shovel.shovel_cycle_time# in mins

        # 按行按列去重复
        loading_time = np.unique(loading_time, axis=0)
        unique_loading_time = np.unique(loading_time, axis=1).astype(int)
        # 对异构铲车的子构数量进行统计
        shovel_type_count = dict()
        for i in range(unique_loading_time.shape[0]):  # truck type index
                int_data = np.array(loading_time[i,:]).astype(int)
                for value in set(int_data):
                    shovel_type_count[f'{i}_{value}'] = list(int_data).count(value)  # it means truck type i w.r.t shovel type num

        # unique_loading_time = np.ones_like(unique_loading_time) + unique_loading_time
        # 对每一行求lcm
        lcm_load_time = np.lcm.reduce(unique_loading_time, axis=1)
        upside_down_sum = 0
        for i in range(unique_loading_time.shape[0]):
            for j in range(unique_loading_time.shape[1]):
                upside_down_sum += shovel_type_count[f'{i}_{unique_loading_time[i,j]}']*(lcm_load_time[i]/unique_loading_time[i,j])
        match_factor = (num_trucks*np.sum(lcm_load_time)) / (upside_down_sum*truck_cycle_time_avg)
        print(f'MatchingFactor: {match_factor}')
        total_wait_time = sum([truck.get_wait_time() for truck in self.mine.trucks])
        print(f'TotalWaitTime: {total_wait_time}')
        # 统计调度算法代码的执行性能
        total_order_count = self.mine.dispatcher.total_order_count
        total_order_time = self.mine.dispatcher.total_order_time
        avg_time_per_order = total_order_time / total_order_count if total_order_count > 0 else 0

        self.ticks["summary"] = {
            "produced_tons":self.ticks[cur_time]['mine_states']['produced_tons'],
            "MatchingFactor":match_factor,
            "TotalWaitTime":total_wait_time,
            "avg_time_per_order":avg_time_per_order,
            "total_order_count":total_order_count,
            "RoadJams":self.ticks[cur_time]['mine_states']['road_jam_count']
        }


    def write_to_file(self, file_name):
        # 替换文件名中的非法字符 ":" 为 "-"
        sanitized_file_name = file_name.replace(":", "-")
        file_path = os.path.join(self.result_path, sanitized_file_name)
        
        with open(file_path, "w") as f:
            try:
                json.dump(self.ticks, f)
                print(f"file_name:{sanitized_file_name} write success")
            except Exception as e:
                print(e)
                print(f"file_name:{sanitized_file_name} write failed")
        return self.ticks


    def read_from_file(self, file_name):
        """
        从文件中读取ticks
        :param file_path:
        :return:
        """
        file_path = os.path.join(self.result_path, file_name)
        with open(file_path, "r") as f:
            try:
                self.ticks = json.load(f)
                print("file_name:{} read success".format(file_name))
            except Exception as e:
                print(e)
                print("file_name:{} read failed".format(file_name))

    def write_truck_states_to_file(self, file_name):
        """
        生成符合新格式的只包含 truck_states 的 JSON 文件。

        :param file_name: 生成文件的文件名
        """
        sanitized_file_name = file_name.replace(":", "-")
        file_path = os.path.join(self.result_path, sanitized_file_name)
        
        # 构造新格式的数据
        truck_states_new_format = []

        for tick, tick_data in self.ticks.items():
            if "truck_states" in tick_data:
                for truck_name, truck_info in tick_data["truck_states"].items():
                    # 提取 timestamp 和 truck_vin 的值
                    timestamp = tick  # 使用 tick 作为 timestamp 值
                    truck_vin = truck_name  # 使用 truck_name 作为 truck_vin 值

                    new_entry = {
                        "command": "truc_scheduling",
                        "signature": "",  # 保留变量名，不赋值
                        "flowId": "",     # 保留变量名，不赋值
                        "timestamp": timestamp,  # 从 tick 获取
                        "data": {
                            "truck_vin": truck_vin,         # 从 truck_name 获取
                            "type": truck_info["state"],   # 之前的 state
                            "dest_code": truck_info["dest_code"]  # 之前的 dest_code
                        }
                    }
                    truck_states_new_format.append(new_entry)

        # 写入新格式的 JSON 文件
        with open(file_path, "w") as f:
            try:
                json.dump(truck_states_new_format, f, indent=4)
                print(f"Truck states file created: {sanitized_file_name}")
            except Exception as e:
                print(e)
                print(f"Failed to write truck states file: {sanitized_file_name}")



                
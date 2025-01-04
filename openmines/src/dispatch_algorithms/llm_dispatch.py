from __future__ import annotations
import numpy as np  # 导入NumPy库
import random,json,time
import openai


from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.charging_site import ChargingSite

from openmines.src.road import Road
from openmines.src.truck import Truck

class LLMDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "LLMDispatcher"
        try:
            self.OPENAI = OPENAI(model_name="gpt-4o")
            print("Successfully initialized OpenAI client")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            raise
        
        self.order_index = 0
        self.init_order_index = 0
        self.common_order_index = 0
        self.init_order_history = []
        self.haul_order_history = []
        self.back_order_history = []
        self.order_history = []
        self.np_random = np.random.RandomState()

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # logger
        self.logger = mine.global_logger.get_logger("LLMDispatcher")
        cur_location = mine.charging_site.name
        
        # 统计loadsite信息
        load_sites = mine.load_sites
        loadsite_queue_length = [load_site.parking_lot.queue_status["total"][int(mine.env.now)] for load_site in load_sites]
        estimated_loadsite_queue_wait_times = [load_site.estimated_queue_wait_time for load_site in load_sites]

        # 获取过去的订单信息
        past_orders_all = self.order_history[-10:]
        past_orders_haul = [order for order in self.order_history if order["order_type"] == "haul_order"][-10:]
        
        # 获取Road距离信息
        road_matrix = mine.road.road_matrix

        # 准备 prompt
        prompt = f"""
                You are now an LLM (Large Language Model) scheduler, and your task is to assign an initial task destination for the current truck based on the available information.
                The current truck is in the charging area and needs to go to the loading area for loading.
                Background knowledge:
                    The mine has multiple loading and unloading areas as well as traffic roads. Mining trucks need to transport goods back and forth between these areas.
                    Loading areas have different loading capacities and queue situations, and unloading areas have different unloading capacities and queue situations.
                    The mining trucks are heterogeneous, varying in their running speeds and loading tonnage.
                    If a road has a large number of mining trucks dispatched, the probability of random events such as traffic jams and road repairs increases, leading to longer operation times for the trucks.

                Current mine information:
                        Loading areas: {[{"name": loadsite.name, "type": "loadsite", "load_capability(tons/min)": loadsite.load_site_productivity, "distance": mine.road.charging_to_load[i],
                                        "queue_length": loadsite_queue_length[i],
                                        "estimated_queue_wait_times": estimated_loadsite_queue_wait_times[i]
                                        } for i, loadsite in enumerate(mine.load_sites)]},

                        Current road information:
                            {[{"road_id": f"{cur_location} to {load_site.name}", "road_desc": f"from {cur_location} to {load_site.name}", "distance": mine.road.charging_to_load[j],
                            "trucks_on_this_road": mine.road.road_status[(cur_location, load_site.name)]["truck_count"],
                            "jammed_trucks_on_this_road": mine.road.road_status[(cur_location, load_site.name)]["truck_jam_count"],
                            "is_road_in_repair": mine.road.road_status[(cur_location, load_site.name)]["repair_count"]} for j, load_site in enumerate(mine.load_sites)]}

                Current order information:
                {{
                "cur_time": {mine.env.now},
                "order_type": "init_order",
                "truck_name": "{truck.name}",
                "truck_capacity": {truck.truck_capacity},
                "truck_speed": {truck.truck_speed}
                }}
                Historical scheduling decisions:
                {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_all]}

                Please assign an initial task destination for the current truck based on the above information,
                Requirements:
                1. Overall objective: Considering the impact of random events on the road, choose the loading area with the shortest distance as much as possible, while avoiding traffic jams and road repairs.
                2. Finally, based on the overall objective, directly provide the following JSON string as the decision result:
                {{
                    "truck_name": "{truck.name}",
                    "loadingsite_index": an integer from 0 to {len(mine.load_sites) - 1}
                }}
                """

        # 初始化变量
        response = "Random selection due to initialization"  # 给一个默认值
        loadsite_index = random.randint(0, len(mine.load_sites) - 1)  # 默认随机值

        try:
            for i in range(3):
                try:
                    tmp_response = self.OPENAI.get_response(prompt=prompt)
                    if tmp_response is None:
                        raise Exception("Empty response from OpenAI")
                    
                    response = tmp_response  # 只在成功时更新 response
                    self.logger.info(f"LLM 订单{self.order_index + 1}：prompt:{prompt} \n {response}")
                    start = response.find('{')
                    if start == -1:
                        raise Exception("No JSON found in response")
                    end = response.rfind('}') + 1
                    if end <= 0:
                        raise Exception("No JSON found in response")
                    
                    json_str = response[start:end]
                    data = json.loads(json_str)
                    loadsite_index = data["loadingsite_index"]
                    break  # 成功获取响应后跳出循环
                except Exception as e:
                    print(f"Attempt {i+1} failed: {e}")
                    self.logger.error(f"LLM 订单{self.order_index + 1}：parse error，giving random order")
                    if i == 2:  # 最后一次尝试也失败了
                        response = f"Random selection due to error after 3 attempts: {str(e)}"
                        loadsite_index = random.randint(0, len(mine.load_sites) - 1)

            # logging
            order = {
                "cur_time": mine.env.now,
                "order_type": "init_order",
                "truck_name": truck.name,
                "truck_capacity": truck.truck_capacity,
                "truck_speed": truck.truck_speed,
                "loadingsite_index": loadsite_index,
                "prompt": prompt,
                "response": response
            }
            
            self.init_order_history.append(order)
            self.order_history.append(order)
            self.order_index += 1
            self.init_order_index += 1
            self.logger.debug(f"LLM INIT 订单{self.init_order_index}：{order}")
            
            return loadsite_index
            
        except Exception as e:
            # 如果出现任何未捕获的异常，确保仍然返回一个有效的结果
            self.logger.error(f"Unexpected error in give_init_order: {e}")
            order = {
                "cur_time": mine.env.now,
                "order_type": "init_order",
                "truck_name": truck.name,
                "truck_capacity": truck.truck_capacity,
                "truck_speed": truck.truck_speed,
                "loadingsite_index": loadsite_index,
                "prompt": prompt,
                "response": f"Random selection due to unexpected error: {str(e)}"
            }
            self.init_order_history.append(order)
            self.order_history.append(order)
            self.order_index += 1
            self.init_order_index += 1
            return loadsite_index

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # logger
        self.logger = mine.global_logger.get_logger("LLMDispatcher")
        # 获取当前卡车信息
        truck_load = truck.truck_load
        cur_location = truck.current_location.name
        cur_loadsite = mine.get_dest_obj_by_name(cur_location)
        cur_loadsite_index = mine.load_sites.index(cur_loadsite)
        assert isinstance(cur_loadsite, LoadSite), f"the truck {truck.name} is not in a loadsite, it is in {cur_loadsite.name}"
        
        # 获取loadsite信息
        load_sites = mine.load_sites
        loadsite_queue_length = [load_site.parking_lot.queue_status["total"][int(mine.env.now)] for load_site in load_sites]
        estimated_loadsite_queue_wait_times = [load_site.estimated_queue_wait_time for load_site in load_sites]

        # 获取dumpsite信息
        avaliable_dumpsites = [dumpsite for dumpsite in mine.dump_sites if dumpsite.parking_lot is not None]
        dump_site_names = [dumpsite.name for dumpsite in avaliable_dumpsites]
        dumpsite_queue_length = [dumpsite.parking_lot.queue_status["total"][int(mine.env.now)] for dumpsite in avaliable_dumpsites]
        estimated_dumpsite_queue_wait_times = [dumpsite.estimated_queue_wait_time for dumpsite in avaliable_dumpsites]

        # 获取过去的订单信息
        past_orders_haul = [order for order in self.order_history if order["order_type"] == "haul_order"][-10:]
        # 获取Road距离信息
        road_matrix = mine.road.road_matrix

        prompt = f"""
                You are now an LLM scheduler, and your task is to assign a target location for the current truck based on the available information.
                Background knowledge:
                The mine has multiple loading and unloading areas as well as traffic roads, and mining trucks need to transport goods back and forth between these areas.
                The loading capacity and queue situation of each loading area are different, as are the unloading capacity and queue situation of each unloading area.
                The mining trucks are heterogeneous, with differences in their running speeds and loading capacities.
                If a road has many mining trucks, the probability of random events such as traffic jams and road maintenance increases, leading to longer operation times for the trucks.

                Current mine information:
                Loading areas: {[{"name": loadsite.name, "type": "loadsite", "load_capability(tons/min)": loadsite.load_site_productivity, "queue_length": loadsite_queue_length[i]
                                } for i, loadsite in enumerate(mine.load_sites)]},
                Unloading areas: {[{"name": dumpsite.name, "type": "dumpsite", "index": j, "distance": road_matrix[cur_loadsite_index][j],
                                    "queue_length": dumpsite_queue_length[j]
                                    } for j, dumpsite in enumerate(avaliable_dumpsites)]},
                Current road information:
                    {[{"road_id": f"{cur_location} to {dumpsite.name}", "road_desc": f"from {cur_location} to {dumpsite.name}", "distance": road_matrix[cur_loadsite_index][j],
                    "trucks_on_this_road": mine.road.road_status[(cur_location, dumpsite.name)]["truck_count"],
                    "jammed_trucks_on_this_road": mine.road.road_status[(cur_location, dumpsite.name)]["truck_jam_count"],
                    "is_road_in_repair": mine.road.road_status[(cur_location, dumpsite.name)]["repair_count"]} for j, dumpsite in enumerate(avaliable_dumpsites)]}

                Historical scheduling decisions:
                    {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_haul]}

                The current truck is at the loading area {cur_location} and needs to go to the unloading area for unloading.
                Current truck order request information:
                {{
                "cur_time": {mine.env.now},
                "order_type": "haul_order",
                "truck_location": "{truck.current_location.name}",
                "truck_name": "{truck.name}",
                "truck_capacity": {truck.truck_capacity},
                "truck_speed": {truck.truck_speed}
                }}

                Please assign a suitable unloading area as the target location for the current truck based on the information above, allowing it to reach the target location as quickly as possible for unloading:
                Requirements:
                1. Overall objective: Considering the impact of random events on the road, choose the unloading area with the shortest distance as much as possible, while avoiding traffic jams and road repairs.
                2. Finally, based on the overall objective, directly provide the following JSON string as the decision result:
                {{
                    "truck_name": "{truck.name}",
                    "dumpsite_index": an integer from 0 to {len(avaliable_dumpsites) - 1}
                }}
                """

        # 初始化 response 和 dumpsite_index
        response = None
        dumpsite_index = None

        for i in range(3):
            try:
                response = self.OPENAI.get_response(prompt)
                if response is None:
                    raise Exception("Empty response from OpenAI")
                    
                self.logger.info(f"LLM 订单{self.order_index + 1}：prompt:{prompt} \n {response}")
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                dumpsite_index = data["dumpsite_index"]
                break
            except Exception as e:
                print(e)
                self.logger.error(f"LLM 订单{self.order_index + 1}：parse error，giving random order")
                if i == 2:  # 最后一次尝试失败
                    dumpsite_index = random.randint(0, len(avaliable_dumpsites) - 1)
                    response = f"Random selection due to error: {str(e)}"

        # 确保有有效的 dumpsite_index
        if dumpsite_index is None:
            dumpsite_index = random.randint(0, len(avaliable_dumpsites) - 1)
            response = "Random selection due to all attempts failed"

        # logging
        order = {
            "cur_time": mine.env.now,
            "order_type": "haul_order",
            "truck_name": truck.name,
            "truck_capacity": truck.truck_capacity,
            "truck_speed": truck.truck_speed,
            "truck_location": f"{truck.current_location.name}",
            "dumpsite_index": dumpsite_index,
            "prompt": prompt,
            "response": response
        }
        self.haul_order_history.append(order)
        self.order_history.append(order)
        self.order_index += 1
        self.logger.debug(f"LLM HAUL 订单{self.order_index}：{order}")
        return dumpsite_index

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # logger
        self.logger = mine.global_logger.get_logger("LLMDispatcher")
        # 获取当前卡车信息
        truck_load = truck.truck_load
        cur_location = truck.current_location.name
        cur_dumpsite = mine.get_dest_obj_by_name(cur_location)
        cur_dumpsite_index = mine.dump_sites.index(cur_dumpsite)
        assert isinstance(cur_dumpsite, DumpSite), f"the truck {truck.name} is not in a dumpsite, it is in {cur_dumpsite.name}"

        # 统计dumpsite信息
        dump_sites = mine.dump_sites
        dumpsite_queue_length = [dump_site.parking_lot.queue_status["total"][int(mine.env.now)] for dump_site in dump_sites]
        estimated_dumpsite_queue_wait_times = [dump_site.estimated_queue_wait_time for dump_site in dump_sites]

        # 获取loadsite信息
        avaliable_loadsites = [loadsite for loadsite in mine.load_sites if loadsite.parking_lot is not None]
        load_site_names = [loadsite.name for loadsite in avaliable_loadsites]
        loadsite_queue_length = [loadsite.parking_lot.queue_status["total"][int(mine.env.now)] for loadsite in avaliable_loadsites]
        estimated_loadsite_queue_wait_times = [loadsite.estimated_queue_wait_time for loadsite in avaliable_loadsites]

        # 获取Road距离信息
        road_matrix = mine.road.road_matrix

        # 历史
        past_orders_back = [order for order in self.order_history if order["order_type"] == "back_order"][-10:]

        prompt = f"""
                    You are now an LLM scheduler, and your task is to assign the current truck a task to return to the loading area.
                    Background knowledge:
                    The mine has multiple loading and unloading areas as well as traffic roads. After being loaded with ore at a loading area, the mining truck needs to go to an unloading area to unload, and then return to the loading area to repeat the process.
                    Each unloading area has different unloading capabilities and queue situations, and each loading area also has different loading capabilities and queue situations.
                    The mining trucks are heterogeneous, with differences in their running speeds and loading capacities.
                    If a road has many mining trucks, the probability of random events like traffic jams and road maintenance increases, leading to longer operation times for the trucks.

                    Current mine information:
                    Unloading areas: {[{"name": dumpsite.name, "type": "dumpsite", "index": i, "queue_length": dumpsite_queue_length[i]
                                        } for i, dumpsite in enumerate(mine.dump_sites)]},
                    Loading areas: {[{"name": loadsite.name, "type": "loadsite", "index": j, "distance": road_matrix[j][cur_dumpsite_index],
                                    "queue_length": loadsite_queue_length[j]
                                    } for j, loadsite in enumerate(avaliable_loadsites)]},
                    Current road information:
                        {[{"road_id": f"{cur_location} to {loadsite.name}", "road_desc": f"from {cur_location} to {loadsite.name}", "distance": road_matrix[j][cur_dumpsite_index],
                            "trucks_on_this_road": mine.road.road_status[(cur_location, loadsite.name)]["truck_count"],
                            "jammed_trucks_on_this_road": mine.road.road_status[(cur_location, loadsite.name)]["truck_jam_count"],
                            "is_road_in_repair": mine.road.road_status[(cur_location, loadsite.name)]["repair_count"]}
                            for j, loadsite in enumerate(avaliable_loadsites)]}

                    Historical scheduling decisions:
                            {[{key: val for key, val in order.items() if key not in ['prompt', 'response']} for order in past_orders_back]}

                    The current truck is at the unloading area {cur_location} and needs to return to the loading area for loading.
                    Current truck order request information:
                    {{
                    "cur_time": {mine.env.now},
                    "order_type": "back_order",
                    "truck_location": "{truck.current_location.name}",
                    "truck_name": "{truck.name}",
                    "truck_capacity": {truck.truck_capacity},
                    "truck_speed": {truck.truck_speed}
                    }}

                    Please assign a suitable loading area as the target location for the current truck based on the information above, allowing it to return as quickly as possible for loading:
                    Requirements:
                        1. Overall objective: Considering the impact of random events on the road, choose the loading area with the shortest distance as much as possible, while avoiding traffic jams and road repairs.
                        2. Finally, based on the overall objective, directly provide the following JSON string as the decision result:
                    {{
                        "truck_name": "{truck.name}",
                        "loadsite_index": an integer number from 0 to {len(avaliable_loadsites) - 1}
                    }}
                """

        # 初始化 response 和 loadsite_index
        response = None
        loadsite_index = None

        for i in range(3):
            try:
                response = self.OPENAI.get_response(prompt)
                if response is None:
                    raise Exception("Empty response from OpenAI")
                    
                self.logger.info(f"LLM 订单{self.order_index + 1}：prompt:{prompt} \n {response}")
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                loadsite_index = data["loadsite_index"]
                break
            except Exception as e:
                print(e)
                self.logger.error(f"LLM 订单{self.order_index + 1}：parse error，giving random order")
                if i == 2:  # 最后一次尝试失败
                    loadsite_index = random.randint(0, len(avaliable_loadsites) - 1)
                    response = f"Random selection due to error: {str(e)}"

        # 确保有有效的 loadsite_index
        if loadsite_index is None:
            loadsite_index = random.randint(0, len(avaliable_loadsites) - 1)
            response = "Random selection due to all attempts failed"

        order = {
            "cur_time": mine.env.now,
            "order_type": "back_order",
            "truck_name": truck.name,
            "truck_capacity": truck.truck_capacity,
            "truck_speed": truck.truck_speed,
            "truck_location": f"{truck.current_location.name}",
            "loadsite_index": loadsite_index,
            "prompt": prompt,
            "response": response
        }
        self.back_order_history.append(order)
        self.order_history.append(order)
        self.order_index += 1
        self.logger.debug(f"LLM BACK 订单{self.order_index}：{order}")
        return loadsite_index
class OPENAI:
    def __init__(self, model_name="gpt-4o"):
        import os
        import openai
        
        # 清除可能存在的环境变量
        os.environ.pop('OPENAI_API_BASE', None)
        os.environ.pop('OPENAI_API_KEY', None)
        
        # 设置 API 配置
        self.api_key = "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"
        self.base_url = "https://api.openai.com/v1"
        self.model_name = model_name

        try:
            # 直接设置 OpenAI 配置
            openai.api_base = self.base_url  # 先设置 base_url
            openai.api_key = self.api_key    # 再设置 api_key
            self._openai = openai
            
            print(f"OpenAI Configuration:")
            print(f"- API base: {self._openai.api_base}")
            print(f"- Model: {self.model_name}")
            print(f"- API key (first 10 chars): {self.api_key[:10]}...")
            
        except Exception as e:
            print(f"Error in OPENAI initialization: {str(e)}")
            print(f"Current environment:")
            print(f"- OPENAI_API_BASE: {os.environ.get('OPENAI_API_BASE')}")
            print(f"- OPENAI_API_KEY: {'Set' if os.environ.get('OPENAI_API_KEY') else 'Not set'}")
            raise

    def get_response(self, prompt):
        if not prompt:
            return "Empty prompt provided"
            
        try:
            # 打印当前配置
            print(f"\nMaking API call with:")
            print(f"- Base URL: {self._openai.api_base}")
            print(f"- Model: {self.model_name}")
            
            # 构造请求
            messages = [{"role": "user", "content": prompt}]
            
            # 发送请求
            response = self._openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 处理响应
            if response and "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"].strip()
            else:
                print("Received empty response from API")
                return "No valid response received"
                
        except Exception as e:
            error_msg = f"API call failed: {str(e)}"
            print(error_msg)
            print(f"Current configuration:")
            print(f"- API base: {self._openai.api_base}")
            print(f"- Model: {self.model_name}")
            return error_msg
        
if __name__ == "__main__":
    from openmines.src.mine import Mine
    from openmines.src.load_site import LoadSite, Shovel, ParkingLot
    from openmines.src.dump_site import DumpSite, Dumper
    from openmines.src.charging_site import ChargingSite
    from openmines.src.road import Road
    from openmines.src.truck import Truck
    import json
    import os
    import numpy as np
    
    def run_test():
        # 配置文件路径
        config_path = r"D:\git clone\openmines\openmines\src\conf\north_pit_mine.json"
        
        try:
            # 加载配置文件
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"成功加载配置文件: {config_path}")
            
            # 创建矿区环境
            mine = Mine(name=config["mine"]["name"])
            print("成功创建矿区环境")
            
            # 创建并添加装载点
            for load_site_config in config["load_sites"]:
                load_site = LoadSite(
                    name=load_site_config["name"],
                    position=tuple(load_site_config["position"])
                )
                
                # 添加铲车
                for i, shovel_config in enumerate(load_site_config["shovels"]):
                    shovel = Shovel(
                        name=shovel_config["name"],
                        shovel_tons=shovel_config["tons"],
                        shovel_cycle_time=shovel_config["cycle_time"],
                        position_offset=tuple(shovel_config["position_offset"])
                    )
                    load_site.add_shovel(shovel)
                
                # 添加停车场并初始化队列状态
                if "parkinglot" in load_site_config:
                    load_site.add_parkinglot(
                        name=load_site_config["parkinglot"]["name"],
                        position_offset=tuple(load_site_config["parkinglot"]["position_offset"])
                    )
                    # 初始化队列状态
                    load_site.parking_lot.queue_status["total"] = {0: 0}
                    for shovel in load_site.shovel_list:
                        load_site.parking_lot.queue_status[shovel.name] = {0: 0}
                    
                mine.add_load_site(load_site)
            print("成功创建装载点")
            
            # 创建并添加卸载点
            for dump_site_config in config["dump_sites"]:
                dump_site = DumpSite(
                    name=dump_site_config["name"],
                    position=tuple(dump_site_config["position"])
                )
                
                # 添加卸载机
                for i, dumper_config in enumerate(dump_site_config["dumpers"]):
                    for j in range(dumper_config["count"]):
                        dumper = Dumper(
                            name=f"{dump_site_config['name']}_Dumper_{j+1}",
                            dumper_cycle_time=dumper_config["cycle_time"],
                            position_offset=tuple(dumper_config["position_offset"])
                        )
                        dump_site.add_dumper(dumper)
                
                # 添加停车场并初始化队列状态
                if "parkinglot" in dump_site_config:
                    dump_site.add_parkinglot(
                        name=dump_site_config["parkinglot"]["name"],
                        position_offset=tuple(dump_site_config["parkinglot"]["position_offset"])
                    )
                    # 初始化队列状态
                    dump_site.parking_lot.queue_status["total"] = {0: 0}
                    for dumper in dump_site.dumper_list:
                        dump_site.parking_lot.queue_status[dumper.name] = {0: 0}
                    
                mine.add_dump_site(dump_site)
            print("成功创建卸载点")
            
            # 创建并添加充电站
            charging_site = ChargingSite(
                name=config["charging_site"]["name"],
                position=tuple(config["charging_site"]["position"])
            )
            
            # 添加卡车
            for truck_type in config["charging_site"]["trucks"]:
                for i in range(truck_type["count"]):
                    truck = Truck(
                        name=f"{truck_type['type']}_{i+1}",
                        truck_capacity=truck_type["capacity"],
                        truck_speed=truck_type["speed"]
                    )
                    charging_site.add_truck(truck)
                    
            mine.add_charging_site(charging_site)
            print("成功创建充电站和卡车")
            
            # 创建并添加道路
            road = Road(
                road_matrix=np.array(config["road"]["road_matrix"]),
                charging_to_load_road_matrix=config["road"]["charging_to_load_road_matrix"],
                road_event_params=config["road"].get("road_event_params")
            )
            
            # 初始化道路状态
            road.road_status = {}
            
            # 初始化充电站到装载点的道路状态
            for load_site in mine.load_sites:
                road.road_status[(config["charging_site"]["name"], load_site.name)] = {
                    "truck_jam_count": 0,
                    "repair_count": 0,
                    "truck_count": 0
                }
            
            # 初始化装载点到卸载点的道路状态
            for load_site in mine.load_sites:
                for dump_site in mine.dump_sites:
                    road.road_status[(load_site.name, dump_site.name)] = {
                        "truck_jam_count": 0,
                        "repair_count": 0,
                        "truck_count": 0
                    }
                    # 反向路径
                    road.road_status[(dump_site.name, load_site.name)] = {
                        "truck_jam_count": 0,
                        "repair_count": 0,
                        "truck_count": 0
                    }
            
            mine.add_road(road)
            print("成功创建道路")
            
            # 创建并添加调度器
            dispatcher = LLMDispatcher()
            mine.add_dispatcher(dispatcher)
            print("成功创建调度器")
            
            
            # 开始测试
            print("\n=== 开始测试单车调度 ===")
            
            # 选择一辆测试卡车
            test_truck = mine.trucks[0]
            test_truck.set_env(mine)
            print(f"选择测试卡车: {test_truck.name}")
            
            # 1. 测试初始调度
            print("\n1. 测试初始调度:")
            init_result = dispatcher.give_init_order(test_truck, mine)
            print(f"初始调度结果: {init_result}")
            
            # 设置卡车位置为目标装载点
            target_load_site = mine.load_sites[init_result]
            test_truck.current_location = target_load_site
            print(f"卡车已移动到装载点: {target_load_site.name}")
            
            # 2. 测试运输调度
            print("\n2. 测试运输调度:")
            haul_result = dispatcher.give_haul_order(test_truck, mine)
            print(f"运输调度结果: {haul_result}")
            
            # 设置卡车位置为目标卸载点
            target_dump_site = mine.dump_sites[haul_result]
            test_truck.current_location = target_dump_site
            print(f"卡车已移动到卸载点: {target_dump_site.name}")
            
            # 3. 测试返回调度
            print("\n3. 测试返回调度:")
            back_result = dispatcher.give_back_order(test_truck, mine)
            print(f"返回调度结果: {back_result}")
            
            # 打印统计信息
            print("\n=== 测试完成 ===")
            print(f"总订单数: {dispatcher.total_order_count}")
            print(f"初始订单数: {dispatcher.init_order_count}")
            
        except Exception as e:
            print(f"\n测试过程中发生错误: {e}")
            import traceback
            print(traceback.format_exc())

    # 运行测试
    run_test()

    
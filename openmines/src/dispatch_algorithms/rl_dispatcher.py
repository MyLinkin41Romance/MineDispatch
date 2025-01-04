from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.load_site import LoadSite
from openmines.src.dump_site import DumpSite

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = x.float()  # 确保输入为float类型
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLDispatcher(BaseDispatcher):
    def __init__(self):
        self.name = "RLDispatcher"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.1

        # 调整状态维度
        # 卡车信息(5) + 装载点信息(3*num_load_sites) + 卸载点信息(3*num_dump_sites)
        self.state_dim = 5 + 3 * 5 + 3 * 5  # 假设5个装载点和5个卸载点
        self.action_dim = 10  # 动作空间保持不变
        
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.criterion = nn.MSELoss()

    def _get_state(self, truck: "Truck", mine: "Mine"):
        state = np.zeros(self.state_dim, dtype=np.float32)  # 明确指定float32类型
        
        # 卡车信息
        if truck.current_location is not None:
            state[0] = 1.0  # 使用浮点数
            if isinstance(truck.current_location, LoadSite):
                state[2] = float(mine.load_sites.index(truck.current_location) + 1)
            elif isinstance(truck.current_location, DumpSite):
                state[2] = float(len(mine.load_sites) + mine.dump_sites.index(truck.current_location) + 1)
            
            state[3:5] = np.array(truck.current_location.position, dtype=np.float32)
        
        # 装载点信息
        offset = 5
        for i, site in enumerate(mine.load_sites):
            base_idx = offset + i * 3
            state[base_idx] = 1.0
            state[base_idx + 1:base_idx + 3] = np.array(site.position, dtype=np.float32)
        
        # 卸载点信息
        offset = offset + len(mine.load_sites) * 3
        for i, site in enumerate(mine.dump_sites):
            base_idx = offset + i * 3
            state[base_idx] = 1.0
            state[base_idx + 1:base_idx + 3] = np.array(site.position, dtype=np.float32)
        
        return torch.FloatTensor(state).to(self.device)


    def _train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.stack([s if len(s.shape) > 1 else s.unsqueeze(0) for s in states]).squeeze(1).float()
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)  # 使用int64类型
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)  # 明确指定float32
        next_states = torch.stack([s if len(s.shape) > 1 else s.unsqueeze(0) for s in next_states]).squeeze(1).float()
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(0.001 * param.data + 0.999 * target_param.data)

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        state = self._get_state(truck, mine)
        
        if random.random() < self.epsilon:
            action = random.randint(0, len(mine.load_sites) - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                # 确保只考虑装载点对应的动作
                valid_actions = q_values[0, :len(mine.load_sites)]
                action = valid_actions.argmax().item()
        
        reward = self._calculate_reward(truck, mine)
        next_state = self._get_state(truck, mine)
        self.memory.append((state, action, reward, next_state))
        
        if len(self.memory) >= self.batch_size:
            self._train()
        
        return action

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        state = self._get_state(truck, mine)
        
        if random.random() < self.epsilon:
            action = random.randint(0, len(mine.dump_sites) - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                # 确保只考虑卸载点对应的动作
                start_idx = len(mine.load_sites)
                end_idx = start_idx + len(mine.dump_sites)
                valid_actions = q_values[0, start_idx:end_idx]
                if valid_actions.numel() > 0:
                    action = valid_actions.argmax().item()
                else:
                    action = random.randint(0, len(mine.dump_sites) - 1)
        
        reward = self._calculate_reward(truck, mine)
        next_state = self._get_state(truck, mine)
        self.memory.append((state, action, reward, next_state))
        
        if len(self.memory) >= self.batch_size:
            self._train()
        
        return action

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        state = self._get_state(truck, mine)
        
        if random.random() < self.epsilon:
            action = random.randint(0, len(mine.load_sites) - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                # 确保只考虑装载点对应的动作
                valid_actions = q_values[0, :len(mine.load_sites)]
                action = valid_actions.argmax().item()
        
        reward = self._calculate_reward(truck, mine)
        next_state = self._get_state(truck, mine)
        self.memory.append((state, action, reward, next_state))
        
        if len(self.memory) >= self.batch_size:
            self._train()
        
        return action

    def _calculate_reward(self, truck: "Truck", mine: "Mine") -> float:
        reward = 0.0
        
        # 基于位置的奖励
        if truck.current_location is not None:
            current_pos = np.array(truck.current_location.position)
            
            if isinstance(truck.current_location, LoadSite):
                reward += 2.0
                # 额外奖励：如果是目标装载点
                if truck.target_location is not None and np.array_equal(current_pos, np.array(truck.target_location.position)):
                    reward += 1.0
            elif isinstance(truck.current_location, DumpSite):
                reward += 1.5
                # 额外奖励：如果是目标卸载点
                if truck.target_location is not None and np.array_equal(current_pos, np.array(truck.target_location.position)):
                    reward += 1.0
            
            # 距离奖励：距离目标位置越近奖励越高
            if truck.target_location is not None:
                target_pos = np.array(truck.target_location.position)
                distance = np.linalg.norm(current_pos - target_pos)
                reward += 2.0 / (1.0 + distance)
        
        # 等待时间惩罚 - 增加惩罚力度
        if hasattr(truck, 'wait_time') and truck.wait_time > 0:
            self.total_wait_time += truck.wait_time
            reward -= 0.3 * truck.wait_time  # 增加等待时间惩罚系数
        
        return reward
if __name__ == "__main__":
    dispatcher = RLDispatcher()
    config_file = "../conf/north_pit_mine.json"
    from openmines.src.mine import Mine
    import json
    import numpy as np
    from openmines.src.charging_site import ChargingSite
    from openmines.src.load_site import LoadSite, Shovel
    from openmines.src.dump_site import DumpSite, Dumper
    from openmines.src.road import Road
    from openmines.src.truck import Truck
    import simpy

    def load_config(filename):
        with open(filename, 'r') as file:
            return json.load(file)

    # 加载配置并初始化矿山
    config = load_config(config_file)
    env = simpy.Environment()  # 创建模拟环境
    mine = Mine(config_file)
    mine.env = env  # 设置环境

    # 初始化充电站和卡车
    charging_site = ChargingSite(config['charging_site']['name'], 
                                position=config['charging_site']['position'])
    trucks = []  # 保存所有卡车的列表，用于测试
    for truck_config in config['charging_site']['trucks']:
        for i in range(truck_config['count']):
            truck = Truck(
                name=f"{truck_config['type']}{i + 1}",
                truck_capacity=truck_config['capacity'],
                truck_speed=truck_config['speed']
            )
            truck.env = env  # 设置环境
            charging_site.add_truck(truck)
            trucks.append(truck)
            mine.trucks.append(truck)  # 将卡车添加到矿山

    # 初始化装载点和铲车
    for load_site_config in config['load_sites']:
        load_site = LoadSite(name=load_site_config['name'], 
                            position=load_site_config['position'])
        load_site.env = env  # 设置环境
        for shovel_config in load_site_config['shovels']:
            shovel = Shovel(
                name=shovel_config['name'],
                shovel_tons=shovel_config['tons'],
                shovel_cycle_time=shovel_config['cycle_time'],
                position_offset=shovel_config['position_offset']
            )
            load_site.add_shovel(shovel)
        load_site.add_parkinglot(
            position_offset=load_site_config['parkinglot']['position_offset'],
            name=load_site_config['parkinglot']['name']
        )
        mine.add_load_site(load_site)

    # 初始化卸载点和卸载机
    for dump_site_config in config['dump_sites']:
        dump_site = DumpSite(dump_site_config['name'], 
                            position=dump_site_config['position'])
        dump_site.env = env  # 设置环境
        for dumper_config in dump_site_config['dumpers']:
            for i in range(dumper_config['count']):
                dumper = Dumper(
                    name=f"{dump_site_config['name']}-{i}",
                    dumper_cycle_time=dumper_config['cycle_time'],
                    position_offset=dumper_config['position_offset']
                )
                dump_site.add_dumper(dumper)
        dump_site.add_parkinglot(
            position_offset=dump_site_config['parkinglot']['position_offset'],
            name=dump_site_config['parkinglot']['name']
        )
        mine.add_dump_site(dump_site)

    # 初始化道路
    road_matrix = np.array(config['road']['road_matrix'])
    road_event_params = config['road'].get('road_event_params', {})
    charging_to_load_road_matrix = config['road']['charging_to_load_road_matrix']
    road = Road(road_matrix=road_matrix, 
                charging_to_load_road_matrix=charging_to_load_road_matrix,
                road_event_params=road_event_params)

    # 添加充电站和道路到矿山
    mine.add_road(road)
    mine.add_charging_site(charging_site)

    # 添加调度器
    mine.add_dispatcher(dispatcher)

    # 开始测试
    print("Testing RL Dispatcher decisions:")
    print("\nInitial orders:")
    for truck in trucks[:3]:  # 测试前3辆卡车
        init_order = dispatcher.give_init_order(truck, mine)
        print(f"Truck {truck.name} initial order: Load Site {init_order}")

    print("\nHaul orders:")
    for truck in trucks[:3]:
        haul_order = dispatcher.give_haul_order(truck, mine)
        print(f"Truck {truck.name} haul order: Dump Site {haul_order}")

    print("\nBack orders:")
    for truck in trucks[:3]:
        back_order = dispatcher.give_back_order(truck, mine)
        print(f"Truck {truck.name} back order: Load Site {back_order}")

    # 测试训练过程
    print("\nTesting training process:")
    for episode in range(5):  # 模拟5个回合
        print(f"\nEpisode {episode + 1}")
        
        # 模拟多次调度决策
        for step in range(3):  # 每个回合进行3次调度
            for truck in trucks[:3]:
                state = dispatcher._get_state(truck, mine)
                
                # 测试不同类型的调度决策
                init_order = dispatcher.give_init_order(truck, mine)
                haul_order = dispatcher.give_haul_order(truck, mine)
                back_order = dispatcher.give_back_order(truck, mine)
                
                # 打印当前状态和决策
                print(f"\nTruck {truck.name} - Step {step + 1}")
                print(f"State: {state.cpu().numpy()}")
                print(f"Orders (Init/Haul/Back): {init_order}/{haul_order}/{back_order}")
                
                # 打印当前的奖励
                reward = dispatcher._calculate_reward(truck, mine)
                print(f"Reward: {reward:.2f}")

        # 打印训练缓冲区大小
        print(f"Memory buffer size: {len(dispatcher.memory)}")

    # 打印最终的训练状态
    print("\nFinal Training Status:")
    print(f"Total production: {dispatcher.total_production}")
    print(f"Total wait time: {dispatcher.total_wait_time}")
    print(f"Final memory buffer size: {len(dispatcher.memory)}")
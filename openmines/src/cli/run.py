import argparse
import json
import pathlib
import sys
import pkgutil
import importlib
from datetime import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from openmines.src.utils.visualization.charter import Charter
from openmines.src.utils.visualization.graphher import VisualGrapher

# add the openmines package to the path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.absolute()))

from openmines.src.mine import Mine
from openmines.src.truck import Truck
from openmines.src.charging_site import ChargingSite
from openmines.src.load_site import LoadSite, Shovel
from openmines.src.dump_site import DumpSite, Dumper
from openmines.src.road import Road
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.dispatch_algorithms.naive_dispatch import NaiveDispatcher
from openmines.src.dispatch_algorithms import *

def load_config(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def run_dispatch_sim(dispatcher: BaseDispatcher, config_file):
    config = load_config(config_file)
    # log_path 为cwd下的logs文件夹
    log_path = pathlib.Path.cwd() / 'logs'
    # 初始化矿山
    mine = Mine(config['mine']['name'], log_path=log_path)
    mine.add_dispatcher(dispatcher)
    # 初始化充电站和卡车
    charging_site = ChargingSite(config['charging_site']['name'], position=config['charging_site']['position'])
    for truck_config in config['charging_site']['trucks']:
        for _ in range(truck_config['count']):
            truck = Truck(
                name=f"{truck_config['type']}{_ + 1}",
                truck_capacity=truck_config['capacity'],
                truck_speed=truck_config['speed']
            )
            charging_site.add_truck(truck)

    # 初始化装载点和铲车
    for load_site_config in config['load_sites']:
        load_site = LoadSite(name=load_site_config['name'], position=load_site_config['position'])
        for shovel_config in load_site_config['shovels']:
            shovel = Shovel(
                name=shovel_config['name'],
                shovel_tons=shovel_config['tons'],
                shovel_cycle_time=shovel_config['cycle_time'],
                position_offset=shovel_config['position_offset']
            )
            load_site.add_shovel(shovel)
        load_site.add_parkinglot(position_offset=load_site_config['parkinglot']['position_offset'],
                                 name=load_site_config['parkinglot']['name'])
        mine.add_load_site(load_site)

    # 初始化卸载点和卸载机
    for dump_site_config in config['dump_sites']:
        dump_site = DumpSite(dump_site_config['name'], position=dump_site_config['position'])
        for dumper_config in dump_site_config['dumpers']:
            for _ in range(dumper_config['count']):
                dumper = Dumper(
                    name=f"{dump_site_config['name']}-{_}",
                    dumper_cycle_time=dumper_config['cycle_time'],
                    position_offset=dumper_config['position_offset']
                )
                dump_site.add_dumper(dumper)
        dump_site.add_parkinglot(position_offset=dump_site_config['parkinglot']['position_offset'],
                                 name=dump_site_config['parkinglot']['name'])
        mine.add_dump_site(dump_site)

    # 初始化道路
    road_matrix = np.array(config['road']['road_matrix'])
    road_event_params = config['road'].get('road_event_params', {})  # 从配置中加载道路事件参数

    charging_to_load_road_matrix = config['road']['charging_to_load_road_matrix']
    road = Road(road_matrix=road_matrix, charging_to_load_road_matrix=charging_to_load_road_matrix,
                road_event_params=road_event_params)
    # # 添加充电站和装载区卸载区
    mine.add_road(road)
    mine.add_charging_site(charging_site)

    # 开始实验
    print(f"Running simulation for {dispatcher.__class__.__name__}")
    ticks = mine.start(total_time=config['sim_time'])
    return ticks


def run_simulation(config_file=None):
    config = load_config(config_file)
    charter = Charter(config_file)
    states_dict = dict()
    
    # [调试信息 1] - 配置文件信息
    print("\n=== Configuration Info ===")
    print(f"Config file: {config_file}")
    print(f"Requested dispatcher types: {config['dispatcher']['type']}")
    
    dispatchers_package = 'openmines.src.dispatch_algorithms'
    dispatchers_module = importlib.import_module(dispatchers_package)
    dispatchers_list = []
    
    # [调试信息 2] - 模块扫描
    print("\n=== Module Scanning ===")
    print(f"Scanning for dispatch algorithms in: {dispatchers_package}")
    
    print("Scanning directory:", dispatchers_module.__path__)

    for _, module_name, _ in pkgutil.iter_modules(dispatchers_module.__path__, dispatchers_package + '.'):
        try:
            # [调试信息 3] - 模块加载
            print(f"\nLoading module: {module_name}")
            module = importlib.import_module(module_name)
            
            # [调试信息 4] - 模块类信息
            module_classes = [x for x in dir(module) if isinstance(getattr(module, x), type)]
            print(f"Classes found in {module_name}: {module_classes}")
            
            dispatcher_types = config['dispatcher']['type']
            for dispatcher_type in dispatcher_types:
                if hasattr(module, dispatcher_type):
                    dispatcher_class = getattr(module, dispatcher_type)
                    dispatcher = dispatcher_class()
                    if dispatcher.name not in [d.name for d in dispatchers_list]:
                        # [调试信息 5] - 调度器添加
                        print(f"Adding dispatcher: {dispatcher.name}")
                        dispatchers_list.append(dispatcher)
                        
        except Exception as e:
            # [调试信息 6] - 错误捕获
            print(f"Error loading module {module_name}: {str(e)}")
    
    # [调试信息 7] - 最终调度器列表
    print("\n=== Dispatcher Summary ===")
    print(f"Found dispatchers: {[d.name for d in dispatchers_list]}")
    
    if not dispatchers_list:
        raise ValueError("No dispatchers found! Check your configuration and dispatcher implementations.")
    
    # [调试信息 8] - 模拟运行信息
    print("\n=== Running Simulations ===")
    for dispatcher in dispatchers_list:
        dispatcher_name = dispatcher.name
        print(f"\nProcessing dispatcher: {dispatcher_name}")
        
        # RUN SIMULATION
        ticks = run_dispatch_sim(dispatcher, config_file)
        print(f"Ticks collected: {len(ticks)}")
        
        # 读取运行结果并保存，等待绘图
        times = []
        produced_tons_list = []
        service_count_list = []
        waiting_truck_count_list = []
        
        for tick in ticks.values():
            if 'mine_states' not in tick:
                continue
            tick = tick['mine_states']
            times.append(tick['time'])
            produced_tons_list.append(tick['produced_tons'])
            service_count_list.append(tick['service_count'])
            waiting_truck_count_list.append(tick['waiting_truck_count'])
        
        # [调试信息 9] - 数据收集结果
        print(f"Data collected for {dispatcher_name}:")
        print(f"- Number of time points: {len(times)}")
        print(f"- Total produced tons: {sum(produced_tons_list)}")
        print(f"- Average service count: {sum(service_count_list)/len(service_count_list) if service_count_list else 0}")
        
        states_dict[dispatcher_name] = {
            'times': times,
            'produced_tons_list': produced_tons_list,
            'service_count_list': service_count_list,
            'waiting_truck_count_list': waiting_truck_count_list,
            'summary': ticks['summary']
        }
    
    # [调试信息 10] - 最终状态字典
    print("\n=== Final States Summary ===")
    print(f"States dictionary keys: {states_dict.keys()}")
    print("Drawing charts...")
    
    # 绘制图表
    charter.draw(states_dict)
    charter.save()
    print("Charts saved successfully!")

def run_visualization(tick_file=None):
    visual_grapher = VisualGrapher(tick_file)
    # 构造路径和文件名字
    gif_file = tick_file.strip('.json') + '.gif'
    visual_grapher.create_animation(output_path=gif_file)

def main():
    parser = argparse.ArgumentParser(description='Run a dispatch simulation of a mine with your DISPATCH algorithm and MINE config file')
    subparsers = parser.add_subparsers(help='commands', dest='command')

    # 直接访问入口 sisymine -f config.json, sisymine -v tick.json
    # 添加运行参数 '-f' 和 '-v'
    parser.add_argument('-f', '--config-file', type=str, help='Path to the config file')
    parser.add_argument('-v', '--tick-file', type=str, help='Path to the simulation tick file')

    # add command 'run'
    run_parser = subparsers.add_parser('run', help='Run a simulation experiment')
    run_parser.add_argument('-f', '--config-file', type=str, required=True, help='Path to the config file')

    # add command visualize
    visualize_parser = subparsers.add_parser('visualize', help='Visualize a simulation experiment')
    visualize_parser.add_argument('-f', '--tick-file', type=str, required=True, help='Path to the simulation tick file')

    args = parser.parse_args()
    # 如command为空，那么检查f/v参数是否存在，如果不存在则print help；如果存在f/v参数则执行run/visualize
    if args.command is None:
        if args.config_file is None and args.tick_file is None:
            parser.print_help()
        elif args.config_file is not None:
            run_simulation(config_file=args.config_file)
        elif args.tick_file is not None:
            run_visualization(tick_file=args.tick_file)
    if args.command == 'run':
        print("args.config_file", args.config_file)
        run_simulation(config_file=args.config_file)
    if args.command == 'visualize':
        tick_file = args.tick_file
        run_visualization(tick_file=tick_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mine simulation')
    parser.add_argument('-f', '--file', type=str, default="D:/git clone/openmines/openmines/src/conf/north_pit_mine.json", help='Path to configuration file')
    args = parser.parse_args()
    
    config_path = args.file
    run_simulation(config_file=config_path)



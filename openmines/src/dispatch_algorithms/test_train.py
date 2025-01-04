# test_train.py
from openmines.src.mine import Mine
import json
import os
from RL_train import train_marl_dispatcher

def load_mine_config():
    config_path = os.path.join('openmines', 'src', 'conf', 'north_pit_mine.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 确保矿区名称是字符串
    if 'mine' not in config:
        config['mine'] = {}
    if 'name' not in config['mine']:
        config['mine']['name'] = "NorthPitMine"
    
    return config

def main():
    # 加载矿区配置
    config = load_mine_config()
    
    # 确认配置内容
    print("Mine configuration:")
    print(f"Mine name: {config['mine']['name']}")
    
    # 创建矿区实例
    mine = Mine(config)
    
    # 使用已有的训练框架进行训练
    print("Starting training phase...")
    network = train_marl_dispatcher(mine)
    
    print("Training completed")

if __name__ == "__main__":
    main()
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.nn import DataParallel

from openmines.src.dispatcher import BaseDispatcher
from openmines.src.mine import Mine
from openmines.src.truck import Truck

class AttentionModel(nn.Module):
    """单独的注意力模型类"""
    def __init__(self, embedding_dim=128, hidden_dim=128, n_heads=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # 嵌入层
        self.load_site_embedder = nn.Linear(3, embedding_dim)
        self.dump_site_embedder = nn.Linear(2, embedding_dim)
        self.truck_embedder = nn.Linear(3, embedding_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim, 1)

class AttentionDispatcher(BaseDispatcher):
    def __init__(self, 
                 embedding_dim=128, 
                 hidden_dim=128, 
                 n_heads=8,
                 n_encode_layers=2):
        super().__init__()
        self.name = "AttentionDispatcher"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建注意力模型实例
        self.attention_model = AttentionModel(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads
        ).to(self.device)

    def compute_embeddings(self, mine: Mine):
        """计算各个点的嵌入"""
        load_sites = mine.load_sites
        dump_sites = mine.dump_sites
        
        # 计算装载点嵌入
        load_site_features = []
        for ls in load_sites:
            productivity = sum(shovel.shovel_tons/shovel.shovel_cycle_time 
                             for shovel in ls.shovel_list)
            features = np.array([*ls.position, productivity])
            load_site_features.append(features)
        
        load_site_features = torch.FloatTensor(load_site_features).to(self.device)
        load_site_embeddings = self.attention_model.load_site_embedder(load_site_features)
        
        # 计算卸载点嵌入
        dump_site_features = []
        for ds in dump_sites:
            features = np.array(ds.position)
            dump_site_features.append(features)
            
        dump_site_features = torch.FloatTensor(dump_site_features).to(self.device)
        dump_site_embeddings = self.attention_model.dump_site_embedder(dump_site_features)
        
        return load_site_embeddings, dump_site_embeddings

    def compute_truck_embedding(self, truck: Truck):
        """计算卡车的嵌入"""
        features = torch.FloatTensor([
            *truck.current_location.position,
            truck.truck_capacity
        ]).to(self.device)
        
        return self.attention_model.truck_embedder(features)

    def select_site(self, 
                    truck_embedding: torch.Tensor,
                    site_embeddings: torch.Tensor) -> int:
        """使用注意力机制选择目标地点"""
        truck_emb = truck_embedding.unsqueeze(0).unsqueeze(0)
        site_emb = site_embeddings.unsqueeze(0)
        
        # 计算注意力
        attn_output, _ = self.attention_model.attention(truck_emb, site_emb, site_emb)
        
        # 计算得分
        scores = self.attention_model.output_layer(attn_output).squeeze()
        
        # 返回得分最高的索引
        return scores.argmax().item()

    def give_init_order(self, truck: Truck, mine: Mine) -> int:
        """初始分配"""
        load_site_embeddings, _ = self.compute_embeddings(mine)
        truck_embedding = self.compute_truck_embedding(truck)
        
        return self.select_site(truck_embedding, load_site_embeddings)

    def give_haul_order(self, truck: Truck, mine: Mine) -> int:
        """选择卸载点"""
        _, dump_site_embeddings = self.compute_embeddings(mine)
        truck_embedding = self.compute_truck_embedding(truck)
        
        return self.select_site(truck_embedding, dump_site_embeddings)

    def give_back_order(self, truck: Truck, mine: Mine) -> int:
        """返回装载点"""
        load_site_embeddings, _ = self.compute_embeddings(mine)
        truck_embedding = self.compute_truck_embedding(truck)
        
        return self.select_site(truck_embedding, load_site_embeddings)

if __name__ == "__main__":
    # 测试代码
    dispatcher = AttentionDispatcher()
    # 这里可以添加与FixedGroupDispatcher相同的测试代码
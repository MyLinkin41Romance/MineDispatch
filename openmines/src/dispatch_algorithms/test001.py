import numpy as np
import matplotlib.pyplot as plt
import random

class EnhancedACO:
    def __init__(self, n_loads, n_dumps, n_trucks, n_ants=20, n_iterations=50, 
                 alpha=1.0, beta=2.0, rho=0.1, q0=0.9):
        self.n_loads = n_loads
        self.n_dumps = n_dumps
        self.n_trucks = n_trucks
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        
        # 设置车队中心点（停车场）位置
        self.depot_position = np.array([50, 50])  # 将车队中心点设在坐标系中心附近
        
        # 随机生成位置
        self.load_positions = np.random.rand(n_loads, 2) * 100
        self.dump_positions = np.random.rand(n_dumps, 2) * 100
        
        # 生成装载点生产能力
        self.load_capacities = np.random.randint(2, 6, size=n_loads)
        
        # 生成卸载点处理能力
        self.dump_capacities = np.random.randint(3, 8, size=n_dumps)
        
        # 计算所有距离矩阵
        self.distance_matrix = self.calculate_distance_matrix()
        self.depot_to_load_distances = self.calculate_depot_distances()
        
        # 初始化信息素矩阵
        self.pheromone = np.ones((n_loads, n_dumps)) * 0.1
        
        # 存储最佳解
        self.best_solution = None
        self.best_truck_assignment = None
        self.best_fitness = float('inf')
        self.best_history = []

    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.n_loads, self.n_dumps))
        for i in range(self.n_loads):
            for j in range(self.n_dumps):
                distance_matrix[i][j] = np.linalg.norm(
                    self.load_positions[i] - self.dump_positions[j]
                )
        return distance_matrix

    def calculate_depot_distances(self):
        # 计算车队中心点到各装载点的距离
        depot_distances = np.zeros(self.n_loads)
        for i in range(self.n_loads):
            depot_distances[i] = np.linalg.norm(
                self.depot_position - self.load_positions[i]
            )
        return depot_distances

    def run(self):
        for iteration in range(self.n_iterations):
            ant_solutions = []
            
            for ant in range(self.n_ants):
                solution, truck_assignment = self.construct_solution()
                fitness = self.calculate_fitness(solution, truck_assignment)
                ant_solutions.append((solution, truck_assignment, fitness))
                
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = solution.copy()
                    self.best_truck_assignment = truck_assignment.copy()
            
            self.best_history.append(self.best_fitness)
            self.update_pheromone(ant_solutions)

    def construct_solution(self):
        solution = np.zeros(self.n_loads, dtype=int)
        truck_assignment = np.zeros(self.n_trucks, dtype=int)
        available_trucks = self.n_trucks
        
        # 考虑从车队中心点出发的距离来分配卡车
        load_priorities = self.depot_to_load_distances.copy()
        load_order = np.argsort(load_priorities)  # 基于到车队中心点距离排序
        
        trucks_assigned = 0
        total_capacity = sum(self.load_capacities)
        
        for load in load_order:
            if random.random() < self.q0:
                dump = self.select_best_dump(load)
            else:
                dump = self.select_dump_roulette(load)
            solution[load] = dump
            
            # 根据装载点能力和到车队中心点的距离分配卡车
            trucks_for_load = int((self.load_capacities[load] / total_capacity) * self.n_trucks)
            if trucks_assigned + trucks_for_load > self.n_trucks:
                trucks_for_load = self.n_trucks - trucks_assigned
            
            for i in range(trucks_for_load):
                if trucks_assigned < self.n_trucks:
                    truck_assignment[trucks_assigned] = load
                    trucks_assigned += 1
        
        return solution, truck_assignment

    def calculate_fitness(self, solution, truck_assignment):
        total_cost = 0
        
        # 计算从车队中心点出发的成本
        for truck in range(self.n_trucks):
            load = truck_assignment[truck]
            dump = solution[load]
            # 计算完整路线距离：车队中心 -> 装载点 -> 卸载点
            route_distance = (self.depot_to_load_distances[load] + 
                            self.distance_matrix[load, dump])
            total_cost += route_distance
        
        # 添加负载平衡惩罚
        for dump in range(self.n_dumps):
            trucks_to_dump = sum(1 for load in range(self.n_loads) 
                               if solution[load] == dump)
            if trucks_to_dump > self.dump_capacities[dump]:
                total_cost += 1000
                
        return total_cost

    def select_best_dump(self, load):
        pheromone_vals = self.pheromone[load]
        distance_vals = self.distance_matrix[load]
        attractiveness = pheromone_vals ** self.alpha * (1.0 / distance_vals) ** self.beta
        return np.argmax(attractiveness)

    def select_dump_roulette(self, load):
        pheromone_vals = self.pheromone[load]
        distance_vals = self.distance_matrix[load]
        attractiveness = pheromone_vals ** self.alpha * (1.0 / distance_vals) ** self.beta
        probabilities = attractiveness / attractiveness.sum()
        return np.random.choice(self.n_dumps, p=probabilities)

    def update_pheromone(self, ant_solutions):
        self.pheromone *= (1 - self.rho)
        
        for solution, truck_assignment, fitness in ant_solutions:
            delta = 1.0 / fitness
            for load in range(self.n_loads):
                dump = solution[load]
                self.pheromone[load, dump] += delta

    def visualize_solution(self):
        plt.figure(figsize=(15, 10))
        
        # 绘制车队中心点
        plt.scatter(self.depot_position[0], self.depot_position[1], 
                   c='green', s=200, marker='*', label='Fleet Depot')
        plt.annotate('Depot', (self.depot_position[0], self.depot_position[1]))
        
        # 绘制装载点
        for i in range(self.n_loads):
            plt.scatter(self.load_positions[i, 0], self.load_positions[i, 1], 
                       c='blue', s=100 + self.load_capacities[i] * 50, 
                       alpha=0.6, label='Loading Sites' if i == 0 else "")
            plt.annotate(f'L{i}\n({self.load_capacities[i]}/h)', 
                        (self.load_positions[i, 0], self.load_positions[i, 1]))
            
            # 绘制从车队中心到装载点的路线
            trucks_at_load = sum(1 for t in self.best_truck_assignment if t == i)
            if trucks_at_load > 0:
                plt.plot([self.depot_position[0], self.load_positions[i][0]],
                        [self.depot_position[1], self.load_positions[i][1]],
                        'k:', alpha=0.3)
        
        # 绘制卸载点
        for i in range(self.n_dumps):
            plt.scatter(self.dump_positions[i, 0], self.dump_positions[i, 1], 
                       c='red', s=100 + self.dump_capacities[i] * 50, 
                       alpha=0.6, label='Dumping Sites' if i == 0 else "")
            plt.annotate(f'D{i}\n({self.dump_capacities[i]}/h)', 
                        (self.dump_positions[i, 0], self.dump_positions[i, 1]))
        
        # 绘制装载点到卸载点的路线
        for load in range(self.n_loads):
            dump = self.best_solution[load]
            trucks_on_route = sum(1 for t in self.best_truck_assignment if t == load)
            
            if trucks_on_route > 0:
                plt.plot([self.load_positions[load][0], self.dump_positions[dump][0]],
                        [self.load_positions[load][1], self.dump_positions[dump][1]],
                        'g--', alpha=0.5)
                mid_x = (self.load_positions[load][0] + self.dump_positions[dump][0]) / 2
                mid_y = (self.load_positions[load][1] + self.dump_positions[dump][1]) / 2
                plt.annotate(f'{trucks_on_route} trucks', (mid_x, mid_y), 
                            bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title('Enhanced ACO Dispatch Solution\nwith Fleet Depot, Trucks and Capacities')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 绘制收敛曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_history)
        plt.title('Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # 创建一个有4个装载点、3个卸载点和10辆卡车的问题
    aco = EnhancedACO(n_loads=4, n_dumps=3, n_trucks=10, n_iterations=50)
    aco.run()
    aco.visualize_solution()
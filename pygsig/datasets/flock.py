import numpy as np
import torch
import yaml

class Flocking(object):
    # TODO
    def __init__(self,
                 config_path='../experiments/config.yaml',
                 data_path='../datasets/flock/data.json'):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            params = config['models']['flock']['simulation']
            self.num_trials = params['num_trials']
            self.x_min = params['x_min']
            self.x_max = params['x_max']
            self.y_min = params['y_min']
            self.y_max = params['y_max']
            self.max_velocity = params['max_velocity']
            self.radius = params['radius']
            self.dist_safe = params['dist_safe']
            self.sigma = params['sigma']
            self.damp_x = params['damp_x']
            self.damp_y = params['damp_y']
            self.num_iterations = params['num_iterations']
            self.dt = params['dt']
            self.num_targets = params['num_targets']
            self.target_seed = params['target_seed']
            self.num_agents = params['num_agents']
            self.min_num_leaders = params['min_num_leaders']
            self.max_num_leaders = params['max_num_leaders']
            self.separation_weight = params['separation_weight']
            self.alignment_weight = params['alignment_weight']
            self.cohesion_weight = params['cohesion_weight']
            self.leader_weight = params['leader_weight']
            self.data_path = data_path
            self.target_loc = np.stack([np.random.uniform(0.25*self.x_min,0.75*self.x_max,size=self.num_targets),
                                        np.random.uniform(0.25*self.y_min,0.75*self.y_max,size=self.num_targets)]).T # locations of targets (fixed)


    def load_data(self,path=None):
        class Agent:
            def __init__(self, x, y, vx, vy, is_leader=False,team=None, target=None, target_id=None):
                self.x = x
                self.y = y
                self.vx = vx
                self.vy = vy
                self.pos = torch.tensor([x,y],dtype=torch.float)
                self.is_leader = is_leader
                self.team = team
                self.target = target
            def dist(self, other):
                return np.linalg.norm([self.x - other.x, self.y - other.y])
        def potential_function(r, dist_safe, epsilon=1e-3):
            # Linear repulsion with cutoff
            if r < dist_safe:
                return 1 / (r + epsilon) - 1 / (dist_safe + epsilon)
            else:
                return 0
        def separation(agent, agents, dist_safe, epsilon=1e-3):
            neighbors = [a for a in agents if a != agent]
            if not neighbors:
                return 0, 0
            sum_fx, sum_fy = 0, 0
            for neighbor in neighbors:
                diff_x = agent.x - neighbor.x
                diff_y = agent.y - neighbor.y
                distance = np.sqrt(diff_x**2 + diff_y**2)
                force_magnitude = potential_function(distance, dist_safe, epsilon)
                sum_fx += force_magnitude * diff_x / (distance+epsilon)
                sum_fy += force_magnitude * diff_y / (distance+epsilon)
            return sum_fx, sum_fy
        def alignment(agent, agents, radius):
            neighbors = [a for a in agents if a != agent and agent.dist(a) < radius]
            if (not neighbors) or (agent.is_leader):
                return agent.vx, agent.vy
            sum_vx, sum_vy = 0, 0
            for neighbor in neighbors:
                sum_vx += neighbor.vx
                sum_vy += neighbor.vy
            return sum_vx / len(neighbors), sum_vy / len(neighbors)
        def cohesion(agent, agents, radius):
            neighbors = [a for a in agents if a != agent and agent.dist(a) < radius]
            if (not neighbors) or (agent.is_leader):
                return 0, 0
            sum_x, sum_y = 0, 0
            for neighbor in neighbors:
                sum_x += neighbor.x
                sum_y += neighbor.y
            avg_x = sum_x / len(neighbors)
            avg_y = sum_y / len(neighbors)
            return avg_x - agent.x, avg_y - agent.y
        def leader_follower(agent, leaders):
            if agent.is_leader:
                return agent.target[0] - agent.x, agent.target[1] - agent.y
            else:
                min_dist = float('inf')
                closest_leader = None
                for leader in leaders:
                    d = agent.dist(leader)
                    if d < min_dist:
                        min_dist = d
                        closest_leader = leader   
                if closest_leader is None:
                    return 0, 0
                return closest_leader.x - agent.x, closest_leader.y - agent.y
        def noise(agent,gain=1):
            return gain*np.random.randn(), gain*np.random.randn()

        def bounce(agent, x_min, x_max, y_min, y_max, err_x, err_y):
            if agent.x < x_min + err_x or agent.x > x_max - err_x:
                agent.vx = -agent.vx
                if agent.x < x_min + err_x:
                    agent.x = x_min + err_x
                if agent.x > x_max - err_x:
                    agent.x = x_max - err_x
            if agent.y < y_min + err_y or agent.y > y_max - err_y:
                agent.vy = -agent.vy
                if agent.y < y_min + err_y:
                    agent.y = y_min + err_y
                if agent.y > y_max - err_y:
                    agent.y = y_max - err_y

        if path is None:
            # Load data from file
            pass
        if path is not None:
            # Generate data
            pass
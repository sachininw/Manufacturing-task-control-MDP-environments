import gym
import numpy as np
from queue import Queue


class InventoryEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=np.array([0]*12),
            high=np.array([1000]*12),
            dtype=np.int64
        )
        self.reset()
        self.seed(5)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=int(seed))

    def demand(self):
        return np.random.poisson(self.demand_)

    def transition(self, s, a, x):
        demand_vec = self.demand()
        self.tot_demand += demand_vec

        arr = np.random.poisson(self.arrivals)

        count_O = list(self.WS_queue[0].queue).count([0, 1, 0])

        occ = [0, 0, 0, 0, 0]
        if self.WS_occupied[0]:
            occ[0] += 1
        if self.WS_occupied[0] and self.WS_inprocess[1] == [0, 1, 0]:
            occ[1] += 1
        if self.WS_occupied[1]:
            occ[2] += 1
        if self.WS_occupied[2]:
            occ[3] += 1
            occ[4] += 1
        if self.WS_occupied[3]:
            occ[1] += 1
            occ[4] += 1

        # [LDQ, DQ, OQ, W1, W2, W3, W4, E, O, L, D, Q]
        self.requirement[0] = max(self.tot_demand[0] - self.WS_queue[0].qsize() - x[7] - occ[0], 0)                             # E
        self.requirement[1] = max(self.tot_demand[1] - count_O - self.WS_queue[3].qsize() - x[8] - occ[1], 0)                   # O
        self.requirement[2] = max(self.tot_demand[2] - self.WS_queue[1].qsize() - x[9] - occ[2], 0)                              # L
        self.requirement[3] = max(self.tot_demand[3] - self.WS_queue[2].qsize() - x[10] - occ[3], 0)                             # D
        self.requirement[4] = max(self.tot_demand[4] - self.WS_queue[2].qsize() - self.WS_queue[3].qsize() - x[11] - occ[4], 0) # Q

        self.WS_queue_copy = self.WS_queue

        release = [[0, 0, 0] for _ in range(4)]

        for i in range(4):
            if self.timestep[i] == self.processing_time[i]:
                release[i] = self.WS_inprocess[i]
                self.WS_inprocess[i] = ''
                self.WS_occupied[i] = False
                self.timestep[i] = 0
                if not self.WS_queue[i].empty():
                    self.WS_inprocess[i] = self.WS_queue[i].get()
                    self.WS_occupied[i] = True
                    self.timestep[i] = 1
            elif self.WS_occupied[i]:
                self.timestep[i] += 1

        actions = [
            [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0],
            [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1],
            [0,0,0,0,0,0]
        ]

        WS_index = 0
        index_start = 0
        k = 0
        repeat = [3, 1, 1, 1]
        index_end = [3, 4, 5, 6]

        for i in repeat:
            for _ in range(i):
                if a == actions[k]:
                    if self.WS_occupied[WS_index]:
                        self.WS_queue[WS_index].put(a[index_start:index_end[WS_index]])
                    else:
                        self.WS_inprocess[WS_index] = a[index_start:index_end[WS_index]]
                        self.WS_occupied[WS_index] = True
                        self.timestep[WS_index] += 1
                k += 1
            index_start = index_end[WS_index]
            WS_index += 1

        x_new = np.array([
            max(x[7] + sum(release[0]) - self.tot_demand[0], 0),
            max(x[8] + release[0][1] + sum(release[3]) - self.tot_demand[1], 0),
            max(x[9] + sum(release[1]) - self.tot_demand[2], 0),
            max(x[10] + sum(release[2]) - self.tot_demand[3], 0),
            max(x[11] + sum(release[2]) + sum(release[3]) - self.tot_demand[4], 0)
        ])

        self.demand_fulfilled = np.array([
            min(self.tot_demand[0], x[7] + sum(release[0])),
            min(self.tot_demand[1], x[8] + release[0][1] + sum(release[3])),
            min(self.tot_demand[2], x[9] + sum(release[1])),
            min(self.tot_demand[3], x[10] + sum(release[2])),
            min(self.tot_demand[4], x[11] + sum(release[2]) + sum(release[3]))
        ])

        self.tot_demand -= self.demand_fulfilled

        for i in range(3):
            if x[i] > 0:
                x[i] -= a[i+3]
            else:
                self.products[i+3] -= a[i+3]
            self.products[i+3] += arr[i+3]

        for i in range(3):
            self.products[i] += arr[i] - a[i]

        self.inventory = np.array([
            max(x[0] + release[0][2], 0),
            max(x[1] + release[1][0], 0),
            max(x[2] + release[0][0], 0),
            self.WS_queue[0].qsize(),
            self.WS_queue[1].qsize(),
            self.WS_queue[2].qsize(),
            self.WS_queue[3].qsize(),
            x_new[0],
            x_new[1],
            x_new[2],
            x_new[3],
            x_new[4]
        ])

        self.state_comp = np.concatenate((self.inventory[0:6], self.requirement), axis=None)

        return self.state_comp, self.demand_fulfilled

    def reward(self, a, requirement):
        tot_inv = np.sum(self.inventory)
        req = [1 if requirement[i] > 0 else 0 for i in range(5)]

        time_penalty = [0] * 4
        time_penalty[0] = self.WS_queue_copy[0].qsize() * 4 * ((a[0]+a[2]) * req[0] + a[1] * max(req[0], req[1]))
        time_penalty[1] = self.WS_queue_copy[1].qsize() * 4 * a[3] * req[2]
        time_penalty[2] = self.WS_queue_copy[2].qsize() * 2 * a[4] * max(req[3], req[4])
        time_penalty[3] = self.WS_queue_copy[3].qsize() * 2 * a[5] * max(req[1], req[4])

        w = [-0.8, -0.7]
        self.Reward = (w[0] * tot_inv) + (w[1] * np.sum(time_penalty))

        return self.Reward

    def step(self, a):
        self.time += 1

        action = [0] * (self.action_space.n - 1)
        if a < self.action_space.n - 1:
            action[a] = 1

        obs_t1, self.demand_fulfilled = self.transition(self.state_comp, action, self.inventory)
        self.state_comp_ = obs_t1
        reward = self.reward(action, obs_t1)
        self.r = reward
        done = self.time == 5000

        return obs_t1, reward, done, {}

    def render(self):
        pass

    def reset(self):
        self.processing_time = [4, 4, 2, 2]
        self.demand_ = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        self.arrivals = np.array([3, 3, 3, 3, 3, 3])
        self.WS_queue = [Queue() for _ in range(4)]
        self.WS_queue_copy = self.WS_queue
        self.WS_inprocess = ['' for _ in range(4)]
        self.timestep = [0 for _ in range(4)]
        self.WS_occupied = [False for _ in range(4)]
        self.tot_demand = np.array([30, 30, 30, 30, 30])
        self.time = 0
        self.Reward = 0
        self.products = [1000, 1000, 1000, 1000, 1000, 1000]
        self.inventory = [0, 2, 0, 1, 1, 5, 0, 0, 2, 0, 1, 1]
        self.requirement = np.array([30, 30, 30, 30, 30])
        self.state_comp = np.concatenate((self.inventory[0:6], self.requirement), axis=None)
        self.action_ = 0

        return self.state_comp

import gym
import numpy as np
from queue import Queue
import os
from sb3_contrib import TRPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback


class InventoryEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.action_space = gym.spaces.Discrete(17)
        # Observation: 19 inventory + 9 requirement + 1 extra = 29
        self.observation_space = gym.spaces.Box(
            low=np.array([0]*29),
            high=np.array([1000]*29),
            dtype=np.int64
        )
        self.reset()
        self.seed(10)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=int(seed))

    def demand(self):
        # BEMOFCQLD
        return np.random.poisson(self.demand_)

    def transition(self, s, a, x):
        self.state_comp_old = self.state_comp

        demand_vec = self.demand()
        # BEMOFCQLD
        self.tot_demand += demand_vec

        arr = np.random.poisson(self.arrivals)

        occ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # WS 0-3: both slots map to the same occ index
        for i in range(4):
            occ[i] += sum(self.WS_occupied[i])
        # WS 4: slot type determines which occ indices to increment
        if self.WS_occupied[4][0] and self.WS_inprocess[4][0] == [1, 0]:
            occ[4] += 1
            occ[5] += 1
        if self.WS_occupied[4][1] and self.WS_inprocess[4][1] == [1, 0]:
            occ[4] += 1
            occ[5] += 1
        if self.WS_occupied[4][0] and self.WS_inprocess[4][0] == [0, 1]:
            occ[4] += 1
            occ[6] += 1
        if self.WS_occupied[4][1] and self.WS_inprocess[4][1] == [0, 1]:
            occ[4] += 1
            occ[6] += 1
        # WS 5
        occ[7] += sum(self.WS_occupied[5])
        # WS 6: increments both occ[8] and occ[6]
        occ[8] += sum(self.WS_occupied[6])
        occ[6] += sum(self.WS_occupied[6])

        count_F = list(self.WS_queue[4].queue).count([1, 0])
        count_C = list(self.WS_queue[4].queue).count([0, 1])

        # BEMOFCQLD
        self.requirement[0] = max(self.tot_demand[0] - self.WS_queue[0].qsize() - x[20] - occ[0], 0)                                          # B
        self.requirement[1] = max(self.tot_demand[1] - self.WS_queue[1].qsize() - x[21] - occ[1], 0)                                          # E
        self.requirement[2] = max(self.tot_demand[2] - self.WS_queue[2].qsize() - x[22] - occ[2], 0)                                          # M
        self.requirement[3] = max(self.tot_demand[3] - self.WS_queue[3].qsize() - x[23] - occ[3], 0)                                          # O
        self.requirement[4] = max(self.tot_demand[4] - count_F - x[24] - occ[4], 0)                                                           # F
        self.requirement[5] = max(self.tot_demand[5] - count_C - x[25] - occ[5], 0)                                                           # C
        self.requirement[6] = max(self.tot_demand[6] - self.WS_queue[4].qsize() - self.WS_queue[6].qsize() - x[26] - occ[6], 0)               # Q
        self.requirement[7] = max(self.tot_demand[7] - self.WS_queue[5].qsize() - x[27] - occ[7], 0)                                          # L
        self.requirement[8] = max(self.tot_demand[8] - self.WS_queue[6].qsize() - x[28] - occ[8], 0)                                          # D

        self.WS_queue_copy = self.WS_queue

        # WS1-4: 4mins each, WS5-6: 2mins, WS7: varies
        release = [[[0, 0, 0], [0, 0, 0]] for _ in range(7)]

        for i in range(7):
            for j in range(2):
                if self.timestep[i][j] == self.processing_time[i]:
                    release[i][j] = self.WS_inprocess[i][j]
                    self.WS_inprocess[i][j] = ''
                    self.WS_occupied[i][j] = False
                    self.timestep[i][j] = 0

                if not self.WS_queue[i].empty():
                    self.WS_inprocess[i][j] = self.WS_queue[i].get()
                    self.WS_occupied[i][j] = True
                    self.timestep[i][j] = 1
                elif self.WS_occupied[i][j]:
                    self.timestep[i][j] += 1

        actions = [
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        ]

        WS_index = 0
        index_start = 0
        k = 0
        repeat = [3, 3, 3, 3, 2, 1, 1]
        index_end = [3, 6, 9, 12, 14, 15, 16]

        for i in repeat:
            for _ in range(i):
                if a == actions[k]:
                    if self.WS_occupied[WS_index][0] and self.WS_occupied[WS_index][1]:
                        self.WS_queue[WS_index].put(a[index_start:index_end[WS_index]])
                    elif not self.WS_occupied[WS_index][0] and self.WS_occupied[WS_index][1]:
                        self.WS_inprocess[WS_index][0] = a[index_start:index_end[WS_index]]
                        self.WS_occupied[WS_index][0] = True
                        self.timestep[WS_index][0] += 1
                    elif self.WS_occupied[WS_index][0] and not self.WS_occupied[WS_index][1]:
                        self.WS_inprocess[WS_index][1] = a[index_start:index_end[WS_index]]
                        self.WS_occupied[WS_index][1] = True
                        self.timestep[WS_index][1] += 1
                    else:
                        self.WS_inprocess[WS_index][0] = a[index_start:index_end[WS_index]]
                        self.WS_occupied[WS_index][0] = True
                        self.timestep[WS_index][0] += 1
                k += 1
            index_start = index_end[WS_index]
            WS_index += 1

        # Fulfil the demand - BEMOFCQLD
        x_new = np.array([
            max(x[20] + sum(release[0][0]) + sum(release[0][1]) - self.tot_demand[0], 0),
            max(x[21] + sum(release[1][0]) + sum(release[1][1]) - self.tot_demand[1], 0),
            max(x[22] + sum(release[2][0]) + sum(release[2][1]) - self.tot_demand[2], 0),
            max(x[23] + sum(release[3][0]) + sum(release[3][1]) - self.tot_demand[3], 0),
            max(x[24] + sum(release[4][0]) + sum(release[4][1]) - self.tot_demand[4], 0),
            max(x[25] + release[4][0][0] + release[4][1][0] - self.tot_demand[5], 0),
            max(x[26] + release[4][0][1] + release[4][1][1] + sum(release[6][0]) + sum(release[6][1]) - self.tot_demand[6], 0),
            max(x[27] + sum(release[5][0]) + sum(release[5][1]) - self.tot_demand[7], 0),
            max(x[28] + sum(release[6][0]) + sum(release[6][1]) - self.tot_demand[8], 0)
        ])

        self.demand_fulfilled = np.array([
            min(self.tot_demand[0], x[20] + sum(release[0][0]) + sum(release[0][1])),
            min(self.tot_demand[1], x[21] + sum(release[1][0]) + sum(release[1][1])),
            min(self.tot_demand[2], x[22] + sum(release[2][0]) + sum(release[2][1])),
            min(self.tot_demand[3], x[23] + sum(release[3][0]) + sum(release[3][1])),
            min(self.tot_demand[4], x[24] + sum(release[4][0]) + sum(release[4][1])),
            min(self.tot_demand[5], x[25] + release[4][0][0] + release[4][1][0]),
            min(self.tot_demand[6], x[26] + release[4][0][1] + release[4][1][1] + sum(release[6][0]) + sum(release[6][1])),
            min(self.tot_demand[7], x[27] + sum(release[5][0]) + sum(release[5][1])),
            min(self.tot_demand[8], x[28] + sum(release[6][0]) + sum(release[6][1]))
        ])

        self.tot_demand -= self.demand_fulfilled

        # state_comp = [ReB, ReE, ReM, ReO, ReF, ReC, ReQ, ReL, ReD, CF, QF, QD]
        for i in range(13):
            if x[i] > 0:
                x[i] -= a[i+3]
            else:
                self.products[i+3] -= a[i+3]
            self.products[i+3] += arr[i+3]

        for i in range(3):
            self.products[i] += arr[i] - a[i]

        self.inventory = np.array([
            max(x[0] + release[0][0][0] + release[0][1][0], 0),
            max(x[1] + release[0][0][1] + release[0][1][1], 0),
            max(x[2] + release[0][0][2] + release[0][1][2], 0),
            max(x[3] + release[1][0][0] + release[1][1][0], 0),
            max(x[4] + release[1][0][1] + release[1][1][1], 0),
            max(x[5] + release[1][0][2] + release[1][1][2], 0),
            max(x[6] + release[2][0][0] + release[2][1][0], 0),
            max(x[7] + release[2][0][1] + release[2][1][1], 0),
            max(x[8] + release[2][0][2] + release[2][1][2], 0),
            max(x[9] + release[3][0][0] + release[3][1][0], 0),
            max(x[10] + release[3][0][2] + release[3][1][2], 0),
            max(x[11] + release[3][0][1] + release[3][1][1], 0),
            max(x[12] + release[5][0][0] + release[5][1][0], 0),
            self.WS_queue[0].qsize(),
            self.WS_queue[1].qsize(),
            self.WS_queue[2].qsize(),
            self.WS_queue[3].qsize(),
            self.WS_queue[4].qsize(),
            self.WS_queue[5].qsize(),
            self.WS_queue[6].qsize(),
            x_new[0],
            x_new[1],
            x_new[2],
            x_new[3],
            x_new[4],
            x_new[5],
            x_new[6],
            x_new[7],
            x_new[8]
        ])

        self.state_comp = np.concatenate((self.inventory[0:19], self.requirement, self.inv), axis=None)

        return self.state_comp, self.demand_fulfilled

    def reward(self, a, state_comp):
        tot_inv = np.sum(self.inventory)
        req = [1 if state_comp[i] > 0 else 0 for i in range(9)]

        time_penalty = [0] * 7
        time_penalty[0] = self.WS_queue_copy[0].qsize() * 4 * (a[0]+a[1]+a[2]) * req[0]
        time_penalty[1] = self.WS_queue_copy[1].qsize() * 3 * (a[3]+a[4]+a[5]) * req[1]
        time_penalty[2] = self.WS_queue_copy[2].qsize() * 4 * (a[6]+a[7]+a[8]) * req[2]
        time_penalty[3] = self.WS_queue_copy[3].qsize() * 2 * (a[9]+a[10]+a[11]) * req[3]
        time_penalty[4] = self.WS_queue_copy[4].qsize() * 2 * (a[12] * max(req[4], req[5]) + a[13] * max(req[5], req[6]))
        time_penalty[5] = self.WS_queue_copy[5].qsize() * 3 * a[14] * req[7]
        time_penalty[6] = self.WS_queue_copy[6].qsize() * 4 * a[15] * max(req[6], req[8])

        w = [-0.5, -1]
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
        self.processing_time = [4, 4, 4, 2, 2, 3, 2]
        self.demand_ = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
        self.arrivals = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.WS_queue = [Queue() for _ in range(7)]
        self.WS_queue_copy = self.WS_queue
        self.WS_inprocess = [['', ''] for _ in range(7)]
        self.timestep = [[0, 0] for _ in range(7)]
        self.WS_occupied = [[False, False] for _ in range(7)]
        self.tot_demand = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10])
        self.time = 0
        self.Reward = 0
        self.products = [1000] * 16
        self.inventory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 6, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.inv = np.array([0])
        self.requirement = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10])
        self.state_comp = np.concatenate((self.inventory[0:19], self.requirement, self.inv), axis=None)
        self.state_comp_old = self.state_comp
        self.action_ = 0

        return self.state_comp


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, mean_reward))
                if mean_reward >= self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
        return True


def make_env(rank, seed=0):
    def _init():
        env = InventoryEnv()
        env.seed(seed + rank)
        env = Monitor(env, filename="TRPOSetup1-8/")
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    savebest_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir="TRPOSetup1-8/")
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path="TRPOSetup1-8/")
    callback = CallbackList([savebest_callback])

    num_cpu = 10
    env = DummyVecEnv([make_env(i) for i in range(num_cpu)])

    model = TRPO.load("/home/weerasekara.s/Setup1/TRPO/TRPOSetup1-8.zip", env)
    model.learn(total_timesteps=50000000, log_interval=50)
    model.save("TRPOSetup1-9")

from agent import Agent, encoder
import time as realtime
from collections import defaultdict
import numpy as np





class Trainer:
    def __init__(self, Env, conf_list):
        self.env = Env(config=conf_list[0])
        self.checkpoint = None
        self.iter = 0

    def train(self, run_time):
        machine_status, job_status, time, job_list = self.env.reset()

        done = False
        lr = 0.0005
        n_games = 500

        agent = Agent(self.env, gamma=0.99, epsilon=1.0, alpha=lr, input_dims=8, n_actions=4, batch_size=64)

        now_time = realtime.time()
        total_time = 0
        while True:
            job_assignment = agent.act(encoder(machine_status, job_status, time, job_list))
            machine_status, job_status, time, reward, job_list, done = self.env.step(job_assignment)
            last_time = now_time
            now_time = realtime.time()
            total_time += now_time - last_time
            self.iter += 1

            if total_time + 2*(now_time-last_time) > run_time:
                break

        return agent






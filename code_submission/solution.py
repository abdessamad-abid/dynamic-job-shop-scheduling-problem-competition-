from agent import Agent
import time as realtime
from collections import defaultdict
import numpy as np
from encoders import *




class Trainer:
    def __init__(self, Env, conf_list):
        self.env = Env(config=conf_list[0])
        self.checkpoint = None
        self.iter = 0

    def train(self, run_time):
        machine_status, job_status, time, job_list = self.env.reset()

        done = False
        state, job_dict, machine_dict = encoder(machine_status, job_status, job_list, self.env, done, time)
        lr = 0.005
        n_games = 1000
        number_of_actions = 1 + len(job_status)

        nb_machines = sum([len(self.env.machines[i]) for i in self.env.machines])
        agent = Agent(self.env,job_dict=job_dict, machine_dict=machine_dict, gamma=0.99, epsilon=1.0, alpha=lr, input_dims=len(state), n_actions=number_of_actions, batch_size=64)

        now_time = realtime.time()
        total_time = 0
        scores = []
        eps_history = []
        for i in range(n_games):
            done = False
            score = 0
            machine_status, job_status, time, job_list = self.env.reset()

            state,job_dict, machine_dict = encoder(machine_status, job_status, job_list, self.env, done, time)
            while not done:
                action =[]
                job_assignment = agent.act(state)
                machine_status, job_status, time, reward, job_list, done = self.env.step(job_assignment)
                new_state, job_dict,machine_dict = encoder(machine_status, job_status, job_list, self.env, done, time)
                score +=reward['makespan']+reward['PTV']
                agent.remember(state,action, score,new_state, done)
                agent.learn()
                last_time = now_time
                now_time = realtime.time()
                total_time += now_time - last_time
                self.iter += 1

                if total_time + 2 * (now_time - last_time) > run_time:
                    break
            if total_time + 2 * (now_time - last_time) > run_time:
                break
            eps_history.append(agent.epsilon_list[i])
            scores.append(score)


        return agent






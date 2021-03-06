from agent import Agent
import time as realtime

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
        lr = 0.01
        n_games = 100
        number_of_actions = 1 + sum([len(self.env.jobs[i]) for i in self.env.jobs])

        agent = Agent(self.env , gamma=0.99, epsilon=1.0, alpha=lr, input_dims=len(state), n_actions=number_of_actions, batch_size=1024)
        agent.machine_status = machine_status
        agent.job_status = job_status
        agent.job_list = job_list
        now_time = realtime.time()
        total_time = 0
        scores = []
        eps_history = []
        for i in range(n_games):
            state, job_dict, machine_dict = encoder(machine_status, job_status, job_list, self.env, done, time)
            done = False
            score = 0
            self.iter = 0
            while not done:
                job_assignment = agent.act(machine_status, job_status, time, job_list)
                agent.machine_status = machine_status
                agent.job_status = job_status
                agent.job_list = job_list
                agent.done = done
                machine_status, job_status, time, reward, job_list, done = self.env.step(job_assignment)
                if (self.iter % 20 == 0):
                    for job in job_status:
                        print("job: ", job, "status: ", job_status[job]['status'])
                new_state, job_dict,machine_dict = encoder(machine_status, job_status, job_list, self.env, done, time)
                score +=reward['makespan']+reward['PTV']
                agent.remember(state,agent.action_list, score,new_state, done)
                #agent.learn()
                state = new_state
                last_time = now_time
                now_time = realtime.time()
                total_time += now_time - last_time
                self.iter += 1
                print('iteration',self.iter,'done:', done)
                if done:
                    print(self.iter, total_time)

                if total_time + 2 * (now_time - last_time) > run_time:
                    break
            if total_time + 2 * (now_time - last_time) > run_time:
                break
            scores.append(score)
            machine_status, job_status, time, job_list = self.env.reset()

        return agent






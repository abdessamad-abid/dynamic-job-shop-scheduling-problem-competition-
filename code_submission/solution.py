from agent import Agent
import time as realtime
from collections import defaultdict
import numpy as np


def encoder(machine_status, job_status, job_list, env_jobs, done ):
    ## machine status one-hot encoded:
    state = []
    types = {'A':[0,0,0,1],
             'B':[0,0,1,0],
             'C':[0,1,0,0],
             'D':[1,0,0,0]}
    for machine in machine_status:
        state += types[machine_status[machine]['type']]

    # machine status one-hot encoded:
    status = {'down':[0,0,1],
              'idle':[0,1,0],
              'work':[1,0,0]}
    for machine in machine_status:
        state += status[machine_status[machine]['status']]


    for machine in machine_status:
        state.append(machine_status[machine]['remain_time'])

    number_of_jobs = 0  #we could have no job to do
    for job_type in env_jobs:
        number_of_jobs += len(job_type)

    for machine in machine_status:
        matrix_job = np.identity(number_of_jobs + 1)
        matrix_job = matrix_job.tolist()
        for i in matrix_job:
            state += i

    for machine in machine_status:

        for job_type in env_jobs:
            for job in job_type:
                if  job in machine_status[machine]['job_list'] :
                    state.append(1)
                else:
                    state.append(0)

    working_state = {'work':[1,0,0,0],
                     'pending':[0,1,0,0],
                     'to_arrive':[0,0,1,0],
                     'done':[0,0,0,1]}

    number_machines = len(machine_status)
    matrix_machines = np.identity(number_machines)
    matrix_machines = matrix_machines.tolist()

    for job in job_status:
        state += working_state[job_status[job]['status']]
        state.append(job_status[job]['priority'])
        if job_status[job]['arrival'] == None:
            state.append(0)
        else:
            state.append(job_status[job]['arrival'])
        state.append(job_status[job]['remain_process_time'])
        state.append(job_status[job]['remain_pending_time'])
        if job_status[job]['machine'] != None:
            state += matrix_machines[int(job_status[job]['machine'][2:]) - 1]
        else:
            state += [0]*number_machines

    for machine in job_list:
        m_list = []
        if job_list[machine] == []:
            state += [0 for i in range(number_of_jobs)]
        else:
            for job in job_status:
                if job in job_list[machine]:
                    m_list.append(1)
                else:
                    m_list.append(0)
        state += m_list
    if done:
        state.append(1)
    else:
        state.append(0)
    return np.array(state)


class Trainer:
    def __init__(self, Env, conf_list):
        self.env = Env(config=conf_list[0])
        self.checkpoint = None
        self.iter = 0

    def train(self, run_time):
        machine_status, job_status, time, job_list = self.env.reset()

        done = False
        state = encoder(machine_status, job_status, job_list, self.env.jobs, done)
        lr = 0.0005
        n_games = 500

        agent = Agent(self.env, gamma=0.99, epsilon=1.0, alpha=lr, input_dims=8, n_actions=4, batch_size=64)

        now_time = realtime.time()
        total_time = 0
        while True:
            job_assignment = agent.act(state)
            machine_status, job_status, time, reward, job_list, done = self.env.step(job_assignment)
            new_state = encoder(machine_status, job_status, job_list, self.env.jobs, done)
            last_time = now_time

            now_time = realtime.time()
            total_time += now_time - last_time
            self.iter += 1

            if total_time + 2*(now_time-last_time) > run_time:
                break

        return agent






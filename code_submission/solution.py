from agent import Agent
import time as realtime
from collections import defaultdict
import numpy as np


def encoder(machine_status, job_status, job_list, env_jobs, done, time ):
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

    number_of_jobs = 0
    for job_type in env_jobs:
        number_of_jobs += len(env_jobs[job_type])

    for machine in machine_status:
        matrix_job = np.identity(number_of_jobs + 1) #number of action is the number of jobs + 1
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

    state.append(time)
    return np.array(state)


class Trainer:
    def __init__(self, Env, conf_list):
        self.env = Env(config=conf_list[0])
        self.checkpoint = None
        self.iter = 0

    def train(self, run_time):
        machine_status, job_status, time, job_list = self.env.reset()

        done = False
        state = encoder(machine_status, job_status, job_list, self.env.jobs, done, time)
        lr = 0.0005
        n_games = 8000
        number_of_actions = 1 + sum([len(self.env.jobs[job_type]) for job_type in self.env.jobs])

        nb_machines = sum([len(self.env.machines[i]) for i in self.env.machines])
        agent = Agent(self.env, gamma=0.99, epsilon=1.0, alpha=lr, input_dims=nb_machines, n_actions=number_of_actions, batch_size=64)

        now_time = realtime.time()
        total_time = 0
        scores = []
        eps_history = []
        for i in range(n_games):
            done = False
            score = 0
            machine_status, job_status, time, job_list = self.env.reset()
            state = encoder(machine_status, job_status, job_list, self.env.jobs, done, time)
            while not done:
                action =[]
                job_assignment = agent.act(state)
                machine_status, job_status, time, reward, job_list, done = self.env.step(job_assignment)
                new_state = encoder(machine_status, job_status, job_list, self.env.jobs, done, time)
                score +=reward['makespan']+reward['PTV']
                for machine in job_assignment:
                    if job_assignment[machine] != None:
                        action.append(int(job_assignment[machine][2:]))
                    else:
                        action.append(0)
                agent.remember(state,action, reward,new_state, done)
                agent.learn()
                last_time = now_time
                now_time = realtime.time()
                total_time += now_time - last_time
                self.iter += 1

                if total_time + 2 * (now_time - last_time) > run_time:
                    break
            eps_history.append(agent.epsilon_list[0])
            scores.append(score)


        return agent






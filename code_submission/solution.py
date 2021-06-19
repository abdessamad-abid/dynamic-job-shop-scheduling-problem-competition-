from agent import Agent
import time as realtime
from collections import defaultdict
import numpy as np


def encoder(machine_status, job_status, job_list, env, done, time ):
    ## machine status one-hot encoded:
    state = []
    types = {}
    #type of machine
    n_types = len(env.machines)
    i = 0
    for type in env.machines:
        T = [0]*n_types
        T[i] = 1
        types[type] = T
        i+=1
    for machine in machine_status:
        state += types[machine_status[machine]['type']]
    #status:
    status = {'down':[1,0,0],
              'idle':[0,1,0],
              'work':[0,0,1]}
    for machine in machine_status:
        state += status[machine_status[machine]['status']]

    #remaning time
    for machine in machine_status:
        state.append(machine_status[machine]['remain_time'])

    #calculate number of jobs:
    number_of_jobs = 0
    for job_type in env.job_types:
        number_of_jobs += len(env.job_types[job_type])

    #create job_dict
    job_dict ={'None':[0]*number_of_jobs}
    i = 0
    for type in env.jobs:
        for job in env.jobs[type]:
            T = [0]*number_of_jobs
            T[i] = 1
            job_dict[job] = T
            i += 1


    for machine in machine_status:
        if machine_status[machine]['job'] != None:
            state += job_dict[machine_status[machine]['job']]
        else:
            state += job_dict['None']

    for machine in machine_status:

        for job_type in env.job_types:
            for job in env.job_types[job_type]:
                if job in machine_status[machine]['job_list'] :
                    state.append(1)
                else:
                    state.append(0)

    working_state = {'work':[1,0,0,0],
                     'pending':[0,1,0,0],
                     'to_arrive':[0,0,1,0],
                     'done':[0,0,0,1]}
    #create machine_dict
    number_machines = len(machine_status)
    machine_dict = {'None':[0]*number_machines}
    i = 0
    for machine in machine_status:
        T = [0]*number_machines
        T[i] = 1
        machine_dict[machine] = T
        i+=1
    #job status
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
            state += machine_dict[job_status[job]['machine']]
        else:
            state += machine_dict['None']
    #job list
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
    return np.array(state), job_dict


class Trainer:
    def __init__(self, Env, conf_list):
        self.env = Env(config=conf_list[0])
        self.checkpoint = None
        self.iter = 0

    def train(self, run_time):
        machine_status, job_status, time, job_list = self.env.reset()

        done = False
        state, job_dict = encoder(machine_status, job_status, job_list, self.env, done, time)
        lr = 0.0005
        n_games = 8000
        number_of_actions = 1 + sum([len(self.env.jobs[job_type]) for job_type in self.env.jobs])

        nb_machines = sum([len(self.env.machines[i]) for i in self.env.machines])
        agent = Agent(self.env,job_dict=job_dict, gamma=0.99, epsilon=1.0, alpha=lr, input_dims=len(state), n_actions=number_of_actions, batch_size=64)

        now_time = realtime.time()
        total_time = 0
        scores = []
        eps_history = []
        for i in range(n_games):
            done = False
            score = 0
            machine_status, job_status, time, job_list = self.env.reset()
            state,job_dict = encoder(machine_status, job_status, job_list, self.env, done, time)
            while not done:
                action =[]
                job_assignment = agent.act(state)
                machine_status, job_status, time, reward, job_list, done = self.env.step(job_assignment)
                new_state, job_dict = encoder(machine_status, job_status, job_list, self.env, done, time)
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
            if total_time + 2 * (now_time - last_time) > run_time:
                break
            eps_history.append(agent.epsilon_list[0])
            scores.append(score)


        return agent






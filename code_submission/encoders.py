import numpy as np


def decode(action_list, job_dict, machine_dict):
    job_assignment = {}
    for i in range(len(action_list)):
        machine = ''
        job = ''
        for m in machine_dict:
            if machine_dict[m][i] == 1:
                machine = m
        if action_list[i] != 0:
            for j in job_dict:
                if job_dict[j][action_list[i]-1] == 1:
                    job = j
        else:
            job = None
        job_assignment[machine] = job
    return job_assignment


def encoder(machine_status, job_status, job_list, env, done, time):
    ## machine status one-hot encoded:
    state = []

    # type of machine
    n_types = len(env.machines)
    types = { 'none': [0]*n_types}
    i = 0
    for type in env.machines:
        T = [0] * n_types
        T[i] = 1
        types[type] = T
        i += 1
    for type in env.machines:
        for machine in env.machines[type]:
            if machine in machine_status:
                state += types[machine_status[machine]['type']]
            else:
                state += types['none']
    # status:
    status = {'down': [1, 0, 0],
              'idle': [0, 1, 0],
              'work': [0, 0, 1],
              'none': [0, 0, 0]}
    for type in env.machines:
        for machine in env.machines[type]:
            if machine in machine_status:
                state += status[machine_status[machine]['status']]
            else:
                state += status['none']

    # remaining time
    for type in env.machines:
        for machine in env.machines[type]:
            if machine in machine_status:
                state.append(machine_status[machine]['remain_time'])
            else:
                state.append(0)
    # calculate number of jobs:
    number_of_jobs = sum([len(env.jobs[i]) for i in env.jobs])

    # create job_dict
    job_dict = {'None': [0] * number_of_jobs}
    i = 0
    for type in env.jobs:
        for job in env.jobs[type]:
            T = [0] * number_of_jobs
            T[i] = 1
            job_dict[job] = T
            i += 1
    for type in env.machines:
        for machine in env.machines[type]:
            if machine in machine_status:
                if machine_status[machine]['job'] != None:
                    state += job_dict[machine_status[machine]['job']]
                else:
                    state += job_dict['None']
            else:
                state += job_dict['None']


    working_state = {'work': [1, 0, 0, 0],
                     'pending': [0, 1, 0, 0],
                     'to_arrive': [0, 0, 1, 0],
                     'done': [0, 0, 0, 1],
                     'none':[0, 0, 0, 0]}
    # create machine_dict
    number_machines = sum([len(env.machines[i]) for i in env.machines])
    machine_dict = {'None': [0] * number_machines}
    i = 0
    for type in env.machines:
        for machine in env.machines[type]:
            if machine in machine_status:
                T = [0] * number_machines
                T[i] = 1
                machine_dict[machine] = T
                i += 1
            else:
                machine_dict[machine] = [0] * number_machines

    # job status
    for type in env.jobs:
        for job in env.jobs[type]:
            if job in job_status:
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

                operation_name = job_status[job]['op']
                for type in env.job_types:
                    for operation in env.job_types[type]:
                        if operation['op_name'] == operation_name:
                            machine_type = operation['machine_type']
                state += types[machine_type]

            else:
                state += working_state['none']
                state.append(0)#priority
                state.append(0)#arrival

                state.append(0)#remaining process time
                state.append(0)#remainig pending time

                state += machine_dict['None']

                state += types['none']


    # job list
    for type in env.machines:
        for machine in env.machines[type]:
            if machine in job_list:
                m_list = []
                if job_list[machine] == []:
                    state += [0] * number_of_jobs
                else:
                    for type in env.jobs:
                        for job in env.jobs[type]:
                            if job in job_list[machine]:
                                m_list.append(1)
                            else:
                                m_list.append(0)
            else:
                m_list = [0] * number_of_jobs
            state += m_list
    if done:
        state.append(1)
    else:
        state.append(0)

    state.append(time)

    return np.array(state), job_dict, machine_dict
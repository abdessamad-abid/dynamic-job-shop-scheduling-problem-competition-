from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):

    #this is not our dqn it's just an exemple

    model = Sequential()
    model.add(Dense(fc1_dims, Activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(fc2_dims, Activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model


def encoder(machine_status, job_status, time, job_list, env_jobs, done ):
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

def  decode(action_list):
    job_assignment = {}
    for i in range(action_list):
        if action_list[i] != 0:
            job_assignment['M0'+str(i+1)] = 'J0'+str(action_list[i])
        else:
            job_assignment['M0' + str(i + 1)] = None
    return job_assignment

class Agent:
    def __init__(self, env, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.996,  epsilon_end=0.01, mem_size=10000):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.env = env
        self.nb_machine=sum([len(env.machines[i]) for i in env.machines])
        self.epsilon_list = [epsilon for i in range(self.nb_machine)]
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory_per_machine = [ReplayBuffer(mem_size, input_dims, n_actions, discrete=True) for i in range(self.nb_machine)]

        self.q_eval_per_machine = [build_dqn(alpha, n_actions, input_dims, 256, 256) for i in range(self.nb_machine)]

    def remember(self, state, list_action, reward, new_state, done):
        for i in range(self.nb_machine):
            self.memory_per_machine[i].store_transition(state, list_action[i], reward, new_state, done)

    def act(self, state):
        state = state[np.newaxis, :]
        action_list=[]
        for i in range(self.nb_machine):
            rand = np.random.random()
            if rand < self.epsilon_list[i]:
                action_list.append(np.random.choice(self.action_space))
            else:
                actions = self.q_eval_per_machine[i].predict(state)
                action_list.append(np.argmax(actions))
        job_assignment = decode(action_list)
        return job_assignment

    def learn(self):
        for i in range(self.nb_machine):
            if self.memory_per_machine[i].mem_cntr > self.batch_size:
                state, action, reward, new_state, done = self.memory_per_machine[i].sample_buffer(self.batch_size)

                action_values = np.array(self.action_space, dtype=np.int8)
                action_indices = np.dot(action, action_values)

                q_eval = self.q_eval_per_machine[i].predict(state)

                q_next = self.q_eval_per_machine[i].predict(new_state)

                q_target = q_eval.copy()

                batch_index = np.arange(self.batch_size, dtype=np.int32)

                q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done

                _ = self.q_eval_per_machine[i].fit(state, q_target, verbose=0)

                self.epsilon_list[i] = self.epsilon_list[i]*self.epsilon_dec if self.epsilon_list[i] > self.epsilon_min else self.epsilon_min

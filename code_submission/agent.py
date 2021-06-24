from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from encoders import *
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

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
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
        new_states = self.new_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, rewards, new_states, done

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):

    model = Sequential()
    model.add(Dense(fc1_dims,input_shape=(input_dims, )))
    model.add(Dropout(0.2))

    model.add(Dense(fc2_dims))
    model.add(Dropout(0.2))

    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model





class Agent:
    def __init__(self, env, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.991,  epsilon_end=0.01, mem_size=1500):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.env = env
        self.machine_status = {}
        self.job_status ={}
        self.job_list = {}
        self.job_assignment = {}
        self.nb_machine=sum([len(env.machines[i]) for i in self.env.machines])
        self.epsilon_list = [epsilon for i in range(self.nb_machine)]
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory_per_machine = [ReplayBuffer(mem_size, input_dims, n_actions, discrete=True) for i in range(self.nb_machine)]
        self.done = False

        self.q_eval_per_machine = [build_dqn(alpha, n_actions, input_dims, 1024, 512) for i in range(self.nb_machine)] #Q_networ for evaluating each machine

    def remember(self, state, list_action, reward, new_state, done):
        for i in range(0,self.nb_machine):
            self.memory_per_machine[i].store_transition(state, list_action[i], reward, new_state, done)

    def act(self, machine_status, job_status, time, job_list):

        state, job_dict, machine_dict = encoder(machine_status, job_status, job_list, self.env, self.done, time)

        self.job_dict = job_dict
        self.machine_dict = machine_dict

        state = state[np.newaxis, :]
        self.action_list=[]
        for i in range(self.nb_machine):
            rand = np.random.random()
            if rand < self.epsilon_list[i]:
                self.action_list.append(np.random.choice(self.action_space))
            else:
                actions = self.q_eval_per_machine[i].predict(state)
                self.action_list.append(np.argmax(actions))
        self.job_assignment = decode(self.action_list, self.job_dict, self.machine_dict, self.machine_status, self.job_status, self.job_list, self.job_assignment)
        return self.job_assignment

    def learn(self):
        for i in range(self.nb_machine):
            if self.memory_per_machine[i].mem_cntr > self.batch_size:
                state, action, reward, new_state, done = self.memory_per_machine[i].sample_buffer(self.batch_size)
                #return the actions from one-hot encoding to numbers
                action_values = np.array(self.action_space, dtype=np.int8)
                action_indices = np.dot(action, action_values)

                q_eval = self.q_eval_per_machine[i].predict(state)

                q_next = self.q_eval_per_machine[i].predict(new_state)

                q_target = q_eval.copy()

                batch_index = np.arange(self.batch_size, dtype=np.int32)

                q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done

                self.q_eval_per_machine[i].fit(state, q_target, verbose=0) #surpress the output

                self.epsilon_list[i] = self.epsilon_list[i]*self.epsilon_dec if self.epsilon_list[i] > self.epsilon_min else self.epsilon_min

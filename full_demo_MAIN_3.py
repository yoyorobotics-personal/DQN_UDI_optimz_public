import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from collections import namedtuple
import time

import torch
import torch.nn as nn
import torch.optim as optim

from  CONFI import *
from CSV_file import *
from PLOT_file import *
from  ENV_class import *
from  DQN_file import *

start_time = time.time()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
if torch.cuda.is_available():
  print("ok cuda")

factor_ranges = [(MIN_A, MAX_A), (MIN_B, MAX_B), (MIN_C, MAX_C)]
# Get number of actions from gym action space
n_actions = 6
# Get the number of state observations
#state, info = env.reset()
n_observations = len(factor_ranges) #环境反馈的值的数量，建议4开始测试

policy_net = DQN(n_observations, n_actions).to(device) #主要的学习函数
target_net = DQN(n_observations, n_actions).to(device) #维持稳定的函数 a check point or milestone
target_net.load_state_dict(policy_net.state_dict()) #用于一开始的同步和周期性的同步

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True) #防止过拟合的优化器
memory = ReplayMemory(MEMORYSIZE)
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward')) #namedtuple 可以写一个有名字的元组,对我而言一般元组就够了,下面单元格展示了它的用途


def optimize_model():
    if len(memory) < BATCH_SIZE: #检查够不够采样，不够直接返回
        return
    transitions = memory.sample(BATCH_SIZE) #采样
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) #转换成独立的列表

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool) #非终止掩码
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state) #合并成单个张量
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch) #计算Q值

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device) #计算所有可能动作的最大 Q 值，并将结果存储
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Main function to run episodes
def run_optimization(file_path, interact_path, num_episodes=NUMPEISODES,
                     factor_ranges=[(MIN_A, MAX_A), (MIN_B, MAX_B), (MIN_C, MAX_C)]):
    # Initialize CSV file
    initialize_csv(file_path)
    initialize_csv(interact_path)
    #previous_mean = 0  # Start without a previous mean for comparison
    mean_values = []
    global countNum
    countNum = 0

    for episode in range(1, num_episodes + 1):
        # Step 1: Initialize the state
        env = CustomEnv(factor_ranges)
        stateMean = env.reset()
        state = stateMean[:-1]
        mean = stateMean[-1]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        global steps_done
        steps_done = 0
        #done = False
        while steps_done < NUMSTEPS :
            sample = random.random()
            #print ("sm", sample)
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done*10) / EPS_DECAY)
            #print("eps",eps_threshold)
            steps_done += 1
            countNum += 1

            if sample > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)

            # Take a step in the environment and get results
            stateMean, reward, done = env.step(steps_done,action,countNum)
            #log_to_csv(file_path, episode, stateMean)
            next_state = stateMean[:-1]
            mean = stateMean[-1]
            #mean_values.append(mean)
            next_state = torch.tensor([next_state], dtype=torch.float32, device=device)
            reward = torch.tensor([reward], device=device)
            #print("step_done",steps_done,"ep",episode)

            # Store the transition in memory
            mean_values.append(mean)
            log_to_csv(file_path, countNum, steps_done, stateMean)
            memory.push(state, action, next_state if reward.item() > 0 else None, reward)
            state = next_state

            # Perform optimization step
            optimize_model()  # No parameters needed; uses global variables
            target_net_state_dict = target_net.state_dict()#软更新
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            # Check if the reward condition is met to end the episode early
            #if reward.item() > 0:
            #   break  # Stop early if reward is achieved

        print("step_done",steps_done,"ep",episode)

        # Plot progress
        #plot_means(mean_values)

    # Display final results
    df = pd.read_csv(file_path)
    max_row = df.loc[df["Mean"].idxmax()]
    max_mean = max_row["Mean"]
    factor_settings = max_row[["Factor_1", "Factor_2", "Factor_3"]]

    last_row = df.iloc[-1]  # This will select the last row in the DataFrame
    last_mean = last_row["Mean"]
    last_factor_settings = last_row[["Factor_1", "Factor_2", "Factor_3"]]

    print("Maximum Mean:", max_mean)
    print("Factor Settings for Maximum Mean:", factor_settings.tolist())
    print("Last Mean:", last_mean)
    print("Factor Settings for Last Mean:", last_factor_settings.tolist())

    # Select the last 10 rows
    last_10_rows = df.iloc[-11:-1]
    max_mean = last_10_rows["Mean"].max()
    max_mean_row = last_10_rows[last_10_rows["Mean"] == max_mean].iloc[0]
    max_mean_factor_settings = max_mean_row[["Factor_1", "Factor_2", "Factor_3"]]
    print("Max Mean in Last 10 Rows:", max_mean)
    print("Factor Settings for Max Mean:",
          max_mean_factor_settings)

    #plt.savefig("plot.png")

# Running the main function
if __name__ == "__main__":
    file_path = FILE_PATH
    interact_path = FILE_PATH_INTERACT
    run_optimization(file_path,interact_path)

#end the gh program
log_factors(FILE_PATH_INTERACT, countNum=None, step=None, factor_1="done", factor_2="done", factor_3="done", mean=None)

#timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")

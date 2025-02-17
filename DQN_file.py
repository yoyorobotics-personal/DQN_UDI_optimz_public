import random
from collections import namedtuple, deque
from CONFI import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward')) #namedtuple 可以写一个有名字的元组,对我而言一般元组就够了,下面单元格展示了它的用途

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) #deque方便的暂存器，用途列下

    def push(self, *args): #添加值
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size): #随机选择的数量
        return random.sample(self.memory, batch_size)

    def __len__(self): #监听长度
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions) #3层和128比较基础的数值，之后可以看它的效果调整

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x): #非线性和梯度传播
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

'''
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:

        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
'''

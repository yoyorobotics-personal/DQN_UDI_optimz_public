BATCH_SIZE = 128 #采样数量64-32
GAMMA = 0.99 #当前行为和之后行为的奖励权重
EPS_START = 0.9 #一开始探索率高
EPS_END = 0.01
EPS_DECAY = 750 #衰减跳到500-750
TAU = 0.005 #更新率可以跳到0.001
LR = 2e-3 #学习率可以保持也可以调高到5e-4

NUMPEISODES = 2
NUMSTEPS = 2
MEMORYSIZE = 2
STEPSIZE = 1

MAX_A, MAX_B, MAX_C = 39, 18, 29
MIN_A, MIN_B, MIN_C = 33, 12, 10

FILE_PATH = "game_data_udi_2.csv"
FILE_PATH_INTERACT = "game_interact_udi_2.csv"
PLOT_AVE = 10

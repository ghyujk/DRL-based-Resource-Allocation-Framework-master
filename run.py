# import tensorflow as tf
import numpy as np
from time import time
from Environment import Env
from DRL import DQN_main, DQN_1, DQN_2, DQN_3, DQN_4

import pandas as pd

# tf.set_random_seed(1)
np.random.seed(1)  # 设置随机种子,替换被舍弃的语句：tf.set_random_seed(1)

MEMORY_CAPACITY = 500
BATCH_SIZE = 32
env = Env.Maze

# csv_file = pd.read_csv('E:\\Rana\\untitled1\\state_his.csv')

def run(): # 基于当前状态（s）选择动作，并根据采取的行动和获得的奖励来学习环境
    np.random.seed(12)
    reward_his_1 = [] # 记录了每步的奖励
    ep_reward = 0
    step = 0
    for i in range(300):
        alpha = 0.9
        c = np.random.normal(0, 1, (4, 2))
        b = np.random.normal(0, 1, (4, 2))
        s = (c * c + b * b) * 0.5 # 初始状态 s
        # 在主函数中，状态 s 被重塑为向量形式，然后被用作代理（RL_main、RL_1、RL_2、RL_3、RL_4）的输入，帮助它们选择动作并与环境交互
        print(s)
        while True:
            s = np.reshape(s, 8)
            action_carrier_select = RL_main.choose_action(s)
            action_1 = RL_1.choose_action(s)
            action_2 = RL_2.choose_action(s)
            action_3 = RL_3.choose_action(s)
            action_4 = RL_4.choose_action(s)
            s = np.reshape(s, (4, 2))
            r_1, rate_1 = env().step_1(s, action_carrier_select, action_1, action_2, action_3, action_4)

            s_ = alpha * s + np.sqrt(1 - alpha ** 2) * np.random.uniform(0, 1, (4, 2))

            step += 1
            s = np.reshape(s, 8)
            s_ = np.reshape(s, 8)
            RL_main.store_transition(s, action_carrier_select, r_1, s_)
            RL_1.store_transition(s, action_1, r_1, s_)
            RL_2.store_transition(s, action_2, r_1, s_)
            RL_3.store_transition(s, action_3, r_1, s_)
            RL_4.store_transition(s, action_4, r_1, s_)

            if (step > 500):
                    RL_main.learn()
                    RL_1.learn()
                    RL_2.learn()
                    RL_3.learn()
                    RL_4.learn()

            s_ = np.reshape(s_, (4, 2))
            s = s_
            reward_his_1.append(rate_1)

            print('-----------------------------------------------------')
            print('DQN_rate', np.mean(reward_his_1))

            if step == 5:
                break
        break
    return reward_his_1

if __name__ == "__main__":
    epochs = 1
    np.random.seed(9)
    # 代码中初始化了五个不同配置的DQN实例,每个对应于不同的代理,训练
    RL_main = DQN_main.DQN_main(env().n_actions_1, env().n_features_1,
                      learning_rate=0.001,
                      reward_decay=0,
                      e_greedy=0.98,
                      replace_target_iter=10,
                      batch_size=32,
                      memory_size=200,
                      epochs=epochs)
    RL_1 = DQN_1.DQN_1(env().n_actions_2, env().n_features_1,
                 learning_rate=0.01,
                 reward_decay=0,
                 e_greedy=0.9,
                 replace_target_iter=5,
                 batch_size=32,
                 memory_size=300,
                 epochs=epochs)
    RL_2 = DQN_2.DQN_2(env().n_actions_2, env().n_features_1,
                 learning_rate=0.01,
                 reward_decay=0,
                 e_greedy=0.9,
                 replace_target_iter=5,
                 batch_size=32,
                 memory_size=300,
                 epochs=epochs)
    RL_3 = DQN_3.DQN_3(env().n_actions_2, env().n_features_1,
                 learning_rate=0.01,
                 reward_decay=0,
                 e_greedy=0.9,
                 replace_target_iter=5,
                 batch_size=32,
                 memory_size=300,
                 epochs=epochs)
    RL_4 = DQN_4.DQN_4(env().n_actions_2, env().n_features_1,
                 learning_rate=0.01,
                 reward_decay=0,
                 e_greedy=0.9,
                 replace_target_iter=5,
                 batch_size=32,
                 memory_size=300,
                 epochs=epochs)

    reward_his_1= run()
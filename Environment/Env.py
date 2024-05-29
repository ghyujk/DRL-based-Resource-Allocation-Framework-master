import numpy as np
from gym import spaces
P_max = 1
P_min = 5
R_min = 0
noise = -124

class Maze():
    def __init__(self):
        self.action_space_1 = [[1, 2, 3, 4], [1, 3, 2, 4], [1, 4, 2, 3], [2, 3, 1, 4], [2, 4, 1, 3], [3, 4, 1, 2]]
        self.action_space_2 = spaces.Box(low=1, high=1, shape=(1,), dtype=np.float32)
        self.n_actions_1 = len(self.action_space_1)
        self.action_space_2 = np.linspace(0, P_max, 10)
        self.n_actions_2 = len(self.action_space_2)
        self.n_features_1 = 8
        self.reward_his = []
        self.reward_random_choice_his = []
        self.reward_max_his = []

    def reset(self):
        distance = np.array([0.1, 0.13, 0.16, 0.18])
        return distance

    def sigmoid(self, x):
        s = (P_max)/2 * x + (P_max)/2
        return s

    def up_link(self, H1, H2, H3, H4, P1, P2, P3, P4):
        if H1 >= H2:
            R1 = np.log(1 + P1 * H1 / (10 ** ((noise - 30) / 10) + P2 * H2))
            R2 = np.log(1 + P2 * H2 / (10 ** ((noise - 30) / 10)))
        else:
            R1 = np.log(1 + P1 * H1 / (10 ** ((noise - 30) / 10)))
            R2 = np.log(1 + P2 * H2 / (10 ** ((noise - 30) / 10) + P1 * H1))
        if H3 >= H4:
            R3 = np.log(1 + P3 * H3 / (10 ** ((noise - 30) / 10) + P4 * H4))
            R4 = np.log(1 + P4 * H4 / (10 ** ((noise - 30) / 10)))
        else:
            R3 = np.log(1 + P3 * H3 / (10 ** ((noise - 30) / 10)))
            R4 = np.log(1 + P4 * H4 / (10 ** ((noise - 30) / 10) + P3 * H3))
        return R1, R2, R3, R4

    # def step(self, a, action_1, action_2):
    #     Action_1 = self.action_space_1[action_1]
    #     P1 = action_2[0]
    #     P2 = action_2[1]
    #     P3 = action_2[2]
    #     P4 = action_2[3]
    #     H1 = a[Action_1[0] - 1, 0]
    #     H2 = a[Action_1[1] - 1, 0]
    #     H3 = a[Action_1[2] - 1, 1]
    #     H4 = a[Action_1[3] - 1, 1]
    #     R1, R2, R3, R4 = Maze().up_link(H1, H2, H3, H4, P1, P2, P3, P4)
    #     reward = R1 + R2 + R3 + R4
    #     rate = (R1/(P1+P_min)) + (R2/(P2+P_min)) + (R3/(P3+P_min)) + (R4/(P4+P_min))
    #     self.reward_his.append(rate)
    #     return reward, rate

    def step_1(self, a, action_1, action_2, action_3, action_4, action_5):
        Action_1 = self.action_space_1[action_1]
        Action_2 = self.action_space_2[action_2]
        Action_3 = self.action_space_2[action_3]
        Action_4 = self.action_space_2[action_4]
        Action_5 = self.action_space_2[action_5]

        P1 = Action_2
        P2 = Action_3
        P3 = Action_4
        P4 = Action_5
        H1 = a[Action_1[0] - 1, 0]
        H2 = a[Action_1[1] - 1, 0]
        H3 = a[Action_1[2] - 1, 1]
        H4 = a[Action_1[3] - 1, 1]
        R1, R2, R3, R4 = Maze().up_link(H1, H2, H3, H4, P1, P2, P3, P4)
        reward = R1 + R2 + R3 + R4
        rate = (R1/(P1+P_min)) + (R2/(P2+P_min)) + (R3/(P3+P_min)) + (R4/(P4+P_min))
        return reward, rate
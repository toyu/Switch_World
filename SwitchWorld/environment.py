import numpy as np
import copy as cp

class Environment(object):
    def __init__(self, row=7, col=7, switch_num=3, switch_positions=((0, 1, 3), (1, 5, 5), (2, 1, 5)), auto_generated=False, obstacle_positions=[]):
        if auto_generated:
            self.Switch_World = np.array([])
            for i in range(switch_num):
                world = np.zeros((row, col))
                np.append(self.Switch_World, world)
            world[switch_positions] = 1
        else:
            world0 = [[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]

            world1 = [[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0]]

            world2 = [[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]

            self.Switch_World = np.array([world0, world1, world2])

        self.switch_positions = list(switch_positions)
        self.switch_num = switch_num
        self.agent_position = [0, 0, 0]
        self.row = row
        self.col = col
        self.reward_flag = False
        if len(obstacle_positions) != 0:
            for pos in obstacle_positions:
                for i in range(switch_num):
                    self.Switch_World[i, pos[0], pos[1]] = 2

    def get_next_state(self, current_state, current_action):
        next_state = cp.deepcopy(current_state)
        if current_action == 0:
            if current_state[1] != 0:
                next_state[1] -= 1
        elif current_action == 1:
            if current_state[1] != self.row-1:
                next_state[1] += 1
        elif current_action == 2:
            if current_state[2] != 0:
                next_state[2] -= 1
        else:
            if current_state[2] != self.col-1:
                next_state[2] += 1

        next_state_tuple = tuple(next_state)
        if self.Switch_World[next_state_tuple] == 1:
            next_state[0] += 1
            if next_state[0] == 3:
                self.reward_flag = True
                next_state[0] = 0
        elif self.Switch_World[next_state_tuple] == 2:
            next_state = current_state

        return next_state

    def get_reward(self):
        if self.reward_flag:
            self.reward_flag = False
            return 1
        else:
            return 0

import numpy as np
import copy as cp


def greedy(values):
    max_values = np.where(values == np.amax(values))
    return np.random.choice(max_values[0])


class Q_learning():
    def __init__(self, learning_rate=0.1, discount_rate=0.9, switch_num=3, row_num=7, col_num=7, action_num=4):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.switch_num = switch_num
        self.row_num = row_num
        self.col_num = col_num
        self.action_num = action_num
        self.Q = np.zeros((switch_num, row_num, col_num, action_num))

    def reset_params(self):
        self.Q = np.zeros_like(self.Q)

    def update_Q(self, current_state, current_action, reward, next_state):
        current_state.append(current_action)
        current_state_action = tuple(current_state)
        next_state = tuple(next_state)

        TD_error = reward + self.discount_rate * np.amax(self.Q[next_state]) - self.Q[current_state_action]
        self.Q[current_state_action] += self.learning_rate * TD_error

    def get_Q(self):
        return self.Q


class Sarsa(Q_learning):
    def update_Q(self, current_state, current_action, reward, next_state):
        TD_error = (reward
                    + self.discount_rate * np.amax(self.Q[next_state])
                    - self.Q[current_state][current_action])
        self.Q[current_state][current_action] += self.learning_rate * TD_error


class Agent():
    def __init__(self, switch_num, row_num, col_num, action_num):
        self.policy = Q_learning(switch_num=switch_num, row_num=row_num, col_num=col_num, action_num=action_num)
        self.switch_num = switch_num
        self.row_num = row_num
        self.col_num = col_num
        self.action_num = action_num

    def set_params_sim(self):
        self.policy.reset_params()

    def set_params_epi(self):
        pass

    def select_action(self):
        pass

    def update(self, current_state, current_action, reward, next_state):
        self.policy.update_Q(current_state, current_action, reward, next_state)


class e_greedy(Agent):
    def __init__(self, e=1.0, delta=1/200, switch_num=3, row_num=7, col_num=7, action_num=4):
        super().__init__(switch_num, row_num, col_num, action_num)
        self.e = e
        self.delta = delta

    def set_params_sim(self):
        super().set_params_sim()
        self.e = 1.0

    def set_params_epi(self):
        self.e -= self.delta

    def select_action(self, current_state):
        if self.e > np.random.rand():
            return np.random.randint(self.action_num)
        else:
            return greedy(self.policy.get_Q()[tuple(current_state)])

    def update(self, current_state, current_action, reward, next_state):
        super().update(current_state, current_action, reward, next_state)


class RS_GRC(Agent):
    def __init__(self, learning_rate=0.1, discount_rate=0.9, switch_num=3, row_num=7, col_num=7, action_num=4):
        super().__init__(switch_num, row_num, col_num, action_num)
        self.r = np.zeros((switch_num, row_num, col_num))
        self.rs = np.zeros((switch_num, row_num, col_num, action_num))
        self.t = np.zeros((switch_num, row_num, col_num, action_num))
        self.t_current = np.zeros((switch_num, row_num, col_num, action_num))
        self.t_post = np.zeros((switch_num, row_num, col_num, action_num))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.rg = 0.06
        self.gamma = 0.9
        self.zeta = 0.05
        self.reward_sum = 0
        self.step = 0
        self.eg = 0
        self.ng = 0

    def set_params_sim(self):
        super().set_params_sim()
        self.r = np.zeros_like(self.r)
        self.rs = np.zeros_like(self.rs)
        self.t = np.zeros_like(self.t)
        self.t_current = np.zeros_like(self.t_current)
        self.t_post = np.zeros_like(self.t_post)
        self.reward_sum = 0
        self.step = 0
        self.eg = 0
        self.ng = 0

    def set_params_epi(self):
        self.reward_sum = 0
        self.step = 0

    def select_action(self, current_state):
        Q = self.policy.get_Q()
        current_state_tuple = tuple(current_state)
        self.rs[current_state_tuple] = self.t[current_state_tuple] * (Q[current_state_tuple] - self.r[current_state_tuple])
        return greedy(self.rs[current_state_tuple])

    def update(self, current_state, current_action, reward, next_state):
        super().update(cp.deepcopy(current_state), current_action, reward, next_state)
        Q = self.policy.get_Q()

        self.reward_sum += reward
        self.step += 1
        etmp = self.reward_sum / self.step
        self.eg = (etmp + self.gamma * (self.ng * self.eg)) / (1 + self.gamma * self.ng)
        self.ng = 1 + self.gamma * self.ng
        delta = min((self.eg - self.rg), 0)
        self.r[tuple(current_state)] = np.amax(Q[tuple(current_state)]) - self.zeta * delta

        action_up = greedy(Q[tuple(next_state)])
        current_state.append(current_action)
        current_state_action = tuple(current_state)
        next_state.append(action_up)
        next_state_action_up = tuple(next_state)

        self.t_post[current_state_action] = (1 - self.learning_rate) * self.t_post[current_state_action] \
                                            + self.learning_rate * self.discount_rate * self.t[next_state_action_up]
        self.t_current[current_state_action] += 1
        self.t[current_state_action] = self.t_current[current_state_action] + self.t_post[current_state_action]

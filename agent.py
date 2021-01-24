import numpy as np
import copy as cp

# greedy法
def greedy(values):
    max_values = np.where(values == np.amax(values))
    return np.random.choice(max_values[0])


# エージェントの親クラス
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

        
# Q学習
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

    # Qテーブルの更新
    def update_Q(self, current_state, current_action, reward, next_state):
        current_state.append(current_action)
        current_state_action = tuple(current_state)
        next_state = tuple(next_state)

        TD_error = reward + self.discount_rate * np.amax(self.Q[next_state]) - self.Q[current_state_action]
        self.Q[current_state_action] += self.learning_rate * TD_error

    def get_Q(self):
        return self.Q

    
# Sarsa
class Sarsa(Q_learning):
    # Qテーブルの更新
    def update_Q(self, current_state, current_action, reward, next_state):
        TD_error = (reward
                    + self.discount_rate * np.amax(self.Q[next_state])
                    - self.Q[current_state][current_action])
        self.Q[current_state][current_action] += self.learning_rate * TD_error


# ε-greedy法
class e_greedy(Agent):
    def __init__(self, e=1.0, delta=1/200, switch_num=3, row_num=7, col_num=7, action_num=4):
        super().__init__(switch_num, row_num, col_num, action_num)
        self.e = e
        self.delta = delta

    # シミュレーションごとにパラメータを初期化
    def set_params_sim(self):
        super().set_params_sim()
        self.e = 1.0

    # エピソードごとにパラメータを初期化
    def set_params_epi(self):
        self.e -= self.delta

    # 行動を選択
    def select_action(self, current_state):
        if self.e > np.random.rand():
            return np.random.randint(self.action_num)
        else:
            return greedy(self.policy.get_Q()[tuple(current_state)])

    # 価値の更新
    def update(self, current_state, current_action, reward, next_state):
        super().update(current_state, current_action, reward, next_state)

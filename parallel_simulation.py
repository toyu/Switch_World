import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from joblib import Parallel, delayed
from time import time
import environment as en
import agent as ag


def simulation(simulation_num, episode_num, agent_num):
    rewards = np.zeros((agent_num, episode_num))
    switch_num = 3
    row_num = 7
    col_num = 7
    action_num = 4

    agent_list = [ag.RS_GRC(switch_num=switch_num, row_num=row_num, col_num=col_num, action_num=action_num),
                  ag.e_greedy(switch_num=switch_num, row_num=row_num, col_num=col_num, action_num=action_num)]

    for sim in range(simulation_num):

        for i, agent in enumerate(agent_list):
            agent.set_params_sim()
            obstacle_flag = False

            for epi in range(episode_num):
                if epi != 0 and epi % 300 == 0:
                    obstacle_flag = not(obstacle_flag)
                if obstacle_flag:
                    switch_world = en.Environment(obstacle_positions=[[5, 3]])
                else:
                    switch_world = en.Environment()

                current_state = [0, 0, 0]
                reward_sum = 0
                agent.set_params_epi()

                for step in range(100):
                    # 行動の選択
                    current_action = agent.select_action(current_state)
                    # 次状態の観測
                    next_state = switch_world.get_next_state(cp.deepcopy(current_state), current_action)
                    # 報酬の観測
                    reward = switch_world.get_reward()
                    reward_sum += reward
                    # 価値の更新
                    agent.update(cp.deepcopy(current_state), current_action, reward, cp.deepcopy(next_state))
                    # 状態の更新
                    current_state = next_state

                rewards[i][epi] += reward_sum

    return rewards


def plot_graph(data, agent_num, data_type_num, episode_num, job_num):
    for i in range(data_type_num):
        graphs = np.zeros((agent_num, episode_num))

        for j in range(job_num):
            if data_type_num == 1:
                graphs += data[j]
            else:
                graphs += data[j][i]

        graphs /= simulation_num
        plt.xlabel('episode')
        plt.ylabel(graph_titles[i])
        plt.ylim([0.0, 8.0])
        # plt.xscale("log")
        for k in range(len(graphs)):
            plt.plot(graphs[k], label=labels[k])
        plt.legend(loc="best")
        plt.savefig(graph_titles[i])
        plt.show()


simulation_num = 1000
job_num = 10
simulation_num_per_job = int(simulation_num / job_num)
episode_num = 1200
agent_num = 2
data_type_num = 1
labels = ["RS_GRC", "QL(e_greedy)"]
graph_titles = ["reward"]

start = time()

data = Parallel(n_jobs=job_num, verbose=10)([delayed(simulation)(simulation_num_per_job, episode_num, agent_num) for i in range(job_num)])
plot_graph(data, agent_num, data_type_num, episode_num, job_num)

print('{}秒かかりました'.format(time() - start))
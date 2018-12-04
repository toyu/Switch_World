import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import environment as en
import agent as ag


def simulation(simulation_num, episode_num):
    labels = ["RS_GRC", "QL(e_greedy)"]
    rewards = np.zeros((len(labels), episode_num))
    switch_num = 3
    row_num = 7
    col_num = 7
    action_num = 4

    agent_list = [ag.RS_GRC(switch_num=switch_num, row_num=row_num, col_num=col_num, action_num=action_num),
                  ag.e_greedy(switch_num=switch_num, row_num=row_num, col_num=col_num, action_num=action_num)]

    for sim in range(simulation_num):
        print(sim + 1)

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

    rewards /= simulation_num
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.ylim([0.0, 8.0])
    for i, graph in enumerate(rewards):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="upper right")
    plt.savefig("2")
    plt.show()


simulation(10, 1200)

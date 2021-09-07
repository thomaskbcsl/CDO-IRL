import numpy as np
import matplotlib.pyplot as plt


def plot_online_regret(results):
    average_results = sum(results[:]) / len(results)
    error_margin_below = np.zeros(np.shape(results)[2])
    for t in range(np.shape(results)[2]):
        for i in range(np.shape(results)[0]):
            error_margin_below[t] = min(average_results[2][t] - results[i][2][t], error_margin_below[t])
    error_margin_above = np.zeros(np.shape(results)[2])
    for t in range(np.shape(results)[2]):
        for i in range(np.shape(results)[0]):
            error_margin_above[t] = max(average_results[2][t] - results[i][2][t], error_margin_above[t])
    plt.style.use('seaborn')
    plt.plot(average_results[2], label="Context-Dependent Online IRL via Linear Programming")
    plt.fill_between(range(np.shape(average_results)[1]), average_results[2] + error_margin_below, average_results[2] + error_margin_above, color='blue', alpha=0.1)
    plt.legend(fontsize=14)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Online Regret", fontsize=14)
    plt.savefig("LP_Maze_Maker_online_regret.pdf", bbox_inches='tight')
    plt.show()


def plot_per_episode_regret(results, max_margin_result):
    average_results = sum(results[:]) / len(results)
    error_margin_below = np.zeros(np.shape(results)[2])
    for t in range(np.shape(results)[2]):
        for i in range(np.shape(results)[0]):
            error_margin_below[t] = min(average_results[1][t] - results[i][1][t], error_margin_below[t])
    error_margin_above = np.zeros(np.shape(results)[2])
    for t in range(np.shape(results)[2]):
        for i in range(np.shape(results)[0]):
            error_margin_above[t] = max(average_results[1][t] - results[i][1][t], error_margin_above[t])
    plt.style.use('seaborn')
    plt.plot(average_results[1][1:31], label="Context-Dependent Online IRL via Linear Programming")
    plt.plot(max_margin_result*np.ones(np.shape(results)[2]), label="Static Environment, Ng. and Russel (2000)")
    plt.fill_between(range(np.shape(average_results)[1]-1), average_results[1][1:31] + error_margin_below[1:31], average_results[1][1:31] + error_margin_above[1:31], color='blue', alpha=0.1)
    plt.legend(fontsize=14)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Per-Episode Regret", fontsize=14)
    plt.savefig("LP_Maze_Maker_per_episode_regret.pdf", bbox_inches='tight')
    plt.show()



results = []
# results.append(np.loadtxt(r"C:\Users\thoma\Desktop\CIRL Experiments\Full Info Opt Dem 30 Episodes Random MDP\0payoff_and_regret.txt"))
results.append(np.loadtxt(r"C:\Users\thoma\Desktop\CIRL Experiments\Full Info Opt Dem 30 Episodes NEW\1_payoff_and_regret.txt"))
results.append(np.loadtxt(r"C:\Users\thoma\Desktop\CIRL Experiments\Full Info Opt Dem 30 Episodes NEW\2_payoff_and_regret.txt"))

print(np.shape(results))
# 0.31776284431292723 - RANDOM MDP
# 0.15604614004887262 - MAZE MAKER
plot_per_episode_regret(results, 0.15604614004887262)
plot_online_regret(results)
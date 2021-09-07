import numpy as np
import matplotlib.pyplot as plt


results = []
results.append(np.loadtxt(r"C:\Users\thoma\Desktop\CIRL Experiments\Full Info Opt Dem 30 Episodes NEW\1_payoff_and_regret.txt"))
results.append(np.loadtxt(r"C:\Users\thoma\Desktop\CIRL Experiments\Full Info Opt Dem 30 Episodes NEW\2_payoff_and_regret.txt"))
# results.append(np.loadtxt(r"C:\Users\thoma\Desktop\Other\0payoff_and_regret.txt"))

average_results = sum(results[:]) / len(results)
# print(average_results)
print(np.shape(results))
# print(len(results))

error_margin_below = np.zeros(np.shape(results)[2])
for t in range(np.shape(results)[2]):
    for i in range(np.shape(results)[0]):
        error_margin_below[t] = min(average_results[2][t] - results[i][2][t], error_margin_below[t])
print("Error-Margin", error_margin_below)
error_margin_above = np.zeros(np.shape(results)[2])
for t in range(np.shape(results)[2]):
    for i in range(np.shape(results)[0]):
        error_margin_above[t] = max(average_results[2][t] - results[i][2][t], error_margin_above[t])
print("Error-Margin Above", error_margin_above)
# loss = 2.1723075174411846 - 1.9916299307347134
# size = 21
# same_policy_max_margin = np.zeros(size)
# for t in range(size-1):
#     same_policy_max_margin[t+1] = average_results[2][1] + t*loss

plt.style.use('seaborn')
plt.plot(average_results[2], label="Context-Dependent Online IRL via Linear Programming")
plt.fill_between(range(np.shape(average_results)[1]), average_results[2] + error_margin_below, average_results[2] + error_margin_above, color='blue', alpha=0.1)
# plt.plot(results[1][2], label="Epsilon")
# plt.plot(same_policy_max_margin)
plt.legend(fontsize=14)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Online Regret", fontsize=14)
# plt.ylim(0, 6.5)
plt.savefig("Linear_Programming_Maze_Maker.pdf", bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import environments
import mdp_solvers
import test_bench_linprog
import helper

# mdp = environments.RandomMDP(n_states=200, n_actions_1=4, n_actions_2=4)
mdp = environments.MazeMaker()

T = 30  # no. of episodes
iterations = 5  # total iterations over which we average

""" The Case of Full Information and Optimal Demonstrations. """

# get optimal policy and return per episode
optimal_policy, optimal_value, optimal_Q = mdp_solvers.value_iteration_MG(mdp)
print("Optimal per Episode Payoff: ", optimal_value[mdp.start_state])
payoff_linprog = np.zeros(T+1)
for i in range(iterations):
    first_policy = helper.get_random_policy(mdp)
    payoff_linprog = test_bench_linprog.test_run_linprog(mdp, T, first_policy)
    regret_linprog = np.zeros(T+1)  # per-episode regret
    for t in range(1, T + 1):
        regret_linprog[t] = optimal_value[mdp.start_state] - payoff_linprog[t]
    cum_regret_linprog = np.zeros(T + 1)  # cumulative regret
    for t in range(1, T + 1):
        cum_regret_linprog[t] = sum(regret_linprog[0:t + 1])
    save = np.array([payoff_linprog, regret_linprog, cum_regret_linprog])
    np.savetxt(str(i)+'payoff_and_regret.txt', save)

    # Plotting results
    plt.style.use('ggplot')
    # plt.plot(payoff_linprog, label="Online per Episode Payoff Algorithm 1")
    # plt.plot(payoff_optimal, label="Optimal per Episode Payoff")
    plt.plot(cum_regret_linprog, label="Cumulative Regret Algorithm 1")
    plt.legend()
    plt.xlabel("Episode")
    plt.savefig(str(i)+"Full-Info-Opt-Dem.pdf")
# # results
# payoff_optimal = np.full(T+1, optimal_value[mdp.start_state])
# payoff_linprog = payoff_linprog / iterations
# regret_linprog = np.zeros(T+1)  # in-episode regret
# for t in range(1, T+1):
#     regret_linprog[t] = optimal_value[mdp.start_state] - payoff_linprog[t]
# cum_regret_linprog = np.zeros(T+1)  # cumulative regret
# for t in range(1, T+1):
#     cum_regret_linprog[t] = sum(regret_linprog[0:t+1])
# print(cum_regret_linprog)

# # save results
# save = np.array([payoff_linprog, regret_linprog, cum_regret_linprog])
# np.savetxt('payoff_and_regret.txt', save)

# """ Plotting Results """
# plt.style.use('ggplot')
# # plt.plot(payoff_linprog, label="Online per Episode Payoff Algorithm 1")
# # plt.plot(payoff_optimal, label="Optimal per Episode Payoff")
# plt.plot(cum_regret_linprog, label="Cumulative Regret Algorithm 1")
# plt.legend()
# plt.xlabel("Episode")
# plt.savefig("Full-Info-Opt-Dem.pdf")

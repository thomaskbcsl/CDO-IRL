import numpy as np
import environments
import helper
import test_bench_linprog
import responses
import mdp_solvers

mdp = environments.MazeMaker()

#mdp = environments.RandomMDP(n_states=200, n_actions_1=4, n_actions_2=4)

optimal_policy, optimal_value, optimal_Q = mdp_solvers.value_iteration_MG(mdp)
print("Optimal per Episode Payoff: ", optimal_value[mdp.start_state])

achieved_value = 0
iterations = 10
for i in range(iterations):
    print(i)
    first_policy = helper.get_random_policy(mdp)
    achieved_value += test_bench_linprog.ng_and_russell(mdp, first_policy)
achieved_value /= iterations
print("Achieved Value", achieved_value)
print(optimal_value[mdp.start_state] - achieved_value)

random_value = 0
for j in range(iterations):
    policy_1 = helper.get_random_policy(mdp)
    policy_2, joint_policy = responses.optimal_response(mdp, policy_1)
    achieved_V, achieved_Q = mdp_solvers.policy_evaluation_MG(mdp, joint_policy)
    random_value += achieved_V[mdp.start_state]
random_value /= iterations
print("Baseline Random", random_value)

import numpy as np
import environments
import helper
import test_bench_linprog
import responses
import mdp_solvers

maze_maker = environments.MazeMaker()

optimal_policy, optimal_value, optimal_Q = mdp_solvers.value_iteration_MG(maze_maker)
print("Optimal per Episode Payoff: ", optimal_value[maze_maker.start_state])

achieved_value = 0
iterations = 50
for i in range(iterations):
    first_policy = helper.get_random_policy(maze_maker)
    achieved_value += test_bench_linprog.ng_and_russell(maze_maker, first_policy)
achieved_value /= iterations
print("Achieved Value", achieved_value)

random_value = 0
for j in range(iterations):
    policy_1 = helper.get_random_policy(maze_maker)
    policy_2, joint_policy = responses.optimal_response(maze_maker, policy_1)
    achieved_V, achieved_Q = mdp_solvers.policy_evaluation_MG(maze_maker, joint_policy)
    random_value += achieved_V[maze_maker.start_state]
random_value /= iterations
print("Baseline Random", random_value)

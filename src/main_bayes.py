import numpy as np
import environments
import mdp_solvers
import approx_planning
import responses
import helper
import test_bench_bayes
mdp = environments.MazeMaker()
mdp.R = mdp.R / sum(mdp.R)
beta = 125
T = 25
width = 1 / 100             # width of the discrete simplex (width-skeleton)
n_iterations = 1
sample_size = 5            # no. of samples from posterior

np.seterr('raise')

""" Notes: 
    1. change back length of trajectories
    2. clarify the prior over beta
    3. clarify the posterior over R and beta """

""" The Case of Partial Information and Boltzmann-rational Demonstrations. """

# Get maximal per episode payoff with knowledge of r when playing optimal joint or using approx value iteration
# Optimal Joint Policy via Centralised Control
opt_joint_policy, V, Q = mdp_solvers.value_iteration_MG(mdp)
opt_joint_policy_1 = np.zeros([mdp.n_states, mdp.n_actions_1])
for s in range(mdp.n_states):
    for a in range(mdp.n_actions_1):
        opt_joint_policy_1[s, a] = sum(opt_joint_policy[s, a, :])

# Approximate Value Iteration for Boltzmann Responses
vi_policy_1, vi_V, vi_Q = approx_planning.approx_value_iteration_boltzmann(mdp, beta)

# Evaluation
response_opt_joint, response_opt_joint_joint = responses.boltzmann_response(mdp, opt_joint_policy_1, beta)
response_vi_policy, response_vi_policy_joint = responses.boltzmann_response(mdp, vi_policy_1, beta)
opt_V, vi_V = mdp_solvers.policy_comparison_MG(mdp, response_opt_joint_joint, response_vi_policy_joint)
print("Optimal Payoff Joint-Opt", opt_V[mdp.start_state])
print("Optimal Payoff Approx VI", vi_V[mdp.start_state])
# results
payoff_optimal_joint = np.full(T+1, opt_V[mdp.start_state])
payoff_optimal_vi = np.full(T+1, vi_V[mdp.start_state])

payoff_bayes = np.zeros(T+1)
for i in range(n_iterations):
    first_policy = helper.get_random_policy(mdp)
    payoff_bayes += test_bench_bayes.test_run_bayes(mdp, beta, width, T, sample_size, first_policy)



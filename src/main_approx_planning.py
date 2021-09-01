import numpy as np
import matplotlib.pyplot as plt
import environments
import mdp_solvers
import approx_planning
import responses

mdp = environments.MazeMaker()
beta = 20
epsilon = 0

""" Boltzmann-Rational Case: """

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
print("J-OPT:", opt_V[mdp.start_state])
print("Value Iteration", vi_V[mdp.start_state])
print("Next Epsilon Greedy... ")

""" Epsilon-Greedy Case: """

# Approximate Value Iteration for Epsilon-Greedy Responses
eps_vi_policy_1, eps_vi_V, eps_vi_Q = approx_planning.approx_value_iteration_eps_greedy(mdp, epsilon)

# Evaluation
eps_response_opt, eps_response_opt_joint = responses.epsilon_greedy_response(mdp, opt_joint_policy_1, epsilon)
eps_response_vi, eps_response_vi_joint = responses.epsilon_greedy_response(mdp, eps_vi_policy_1, epsilon)

print('Value Function')
eps_opt_V, eps_vi_V = mdp_solvers.policy_comparison_MG(mdp, eps_response_opt_joint, eps_response_vi_joint)
print(eps_opt_V[mdp.start_state])
print(eps_vi_V[mdp.start_state])

""" Plots:
    create plots where we change the beta and epsilon. """

# Boltzmann:
joint_pol_value = []
vi_value = []
betas = 21
for b in range(betas):
    print("Beta is", b)
    # approximate Value Iteration for Boltzmann Responses
    vi_policy_1, vi_V, vi_Q = approx_planning.approx_value_iteration_boltzmann(mdp, b)
    # responses w.r.t. optimal-joint and approx. value iteration
    response_opt_joint, response_opt_joint_joint = responses.boltzmann_response(mdp, opt_joint_policy_1, b)
    response_vi_policy, response_vi_policy_joint = responses.boltzmann_response(mdp, vi_policy_1, b)
    opt_V, vi_V = mdp_solvers.policy_comparison_MG(mdp, response_opt_joint_joint, response_vi_policy_joint)
    joint_pol_value.append(opt_V[mdp.start_state])
    vi_value.append(vi_V[mdp.start_state])

# Plot Boltzmann
plt.style.use('ggplot')
plt.figure(0)
plt.plot(joint_pol_value, label="Commitment of Optimal Joint Policy")
plt.plot(vi_value, label="Approx. Value Iteration")
plt.legend()
plt.xlabel("Beta")
plt.ylabel("Return")
plt.savefig("Boltzmann_Value_Iteration_Comp.pdf", bbox_inches='tight')

# epsilon-greedy
eps_joint_pol_values = []
eps_vi_values = []
steps = 21
for eps in np.linspace(0, 1, steps):
    print("Epsilon is", eps)
    # Approximate Value Iteration for Epsilon-Greedy Responses
    eps_vi_policy_1, eps_vi_V, eps_vi_Q = approx_planning.approx_value_iteration_eps_greedy(mdp, eps)
    # responses
    eps_response_opt, eps_response_opt_joint = responses.epsilon_greedy_response(mdp, opt_joint_policy_1, eps)
    eps_response_vi, eps_response_vi_joint = responses.epsilon_greedy_response(mdp, eps_vi_policy_1, eps)
    eps_opt_V, eps_vi_V = mdp_solvers.policy_comparison_MG(mdp, eps_response_opt_joint, eps_response_vi_joint)
    eps_joint_pol_values.insert(0, eps_opt_V[mdp.start_state])
    eps_vi_values.insert(0, eps_vi_V[mdp.start_state])

# Plot epsilon-greedy
plt.style.use('ggplot')
plt.figure(1)
plt.plot(np.linspace(0, 1, steps), eps_joint_pol_values, label="Commitment of Optimal Joint Policy")
plt.plot(np.linspace(0, 1, steps), eps_vi_values, label="Approx. Value Iteration")
plt.legend()
plt.xlabel("1-Epsilon")
plt.ylabel("Return")
plt.savefig("Eps_Greedy_Value_Iteration_Comp.pdf", bbox_inches='tight')
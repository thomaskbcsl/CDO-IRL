import numpy as np
import environments
import responses
import helper

mdp = environments.RandomMDP(n_states=200, n_actions_1=4, n_actions_2=4)


beta = 10

policy_1 = helper.get_random_policy(mdp)
policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1, beta)

for i in range(10):
    print(np.round(policy_2[i, :], 2))



#####
print(sum(mdp.R))
A = sum(mdp.R)
mdp.R = mdp.R / sum(mdp.R)
beta = 10 *A
policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1, beta)

for i in range(10):
    print(np.round(policy_2[i, :], 2))




#
print("...............")

mdp = environments.MazeMaker()

beta = 10
policy_1 = helper.get_random_policy(mdp)
policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1, beta)

for i in range(10):
    print(np.round(policy_2[i, :], 2))

print(sum(mdp.R))
A = sum(mdp.R)
mdp.R = mdp.R / sum(mdp.R)
beta = beta * A
policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1, beta)

for i in range(10):
    print(np.round(policy_2[i, :], 2))
print(np.round(policy_2[24, :], 2))

print(20 * 24)
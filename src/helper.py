import numpy as np


""" Helping Functions """

# get a random deterministic policy
def get_random_policy(mdp):
    policy = np.zeros([mdp.n_states, mdp.n_actions_1])
    for s in range(mdp.n_states):
        policy[s][np.random.choice(mdp.n_actions_1, 1)] = 1
    return policy
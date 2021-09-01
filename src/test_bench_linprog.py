import numpy as np
from scipy.optimize import linprog
import itertools
import copy
import mdp_solvers
import responses
import environments
import helper

""" Algorithm 1 """


# test run for Full Information and Optimal Demonstrations
def test_run_linprog(mdp, n_episodes, first_policy):
    # per episode return
    payoff = [0]

    # initialise linear program
    constraint_matrix = np.ones([1, mdp.n_states]) * -1  # positive values (place holder)
    A_eq = np.ones([1, mdp.n_states])  # simplex
    b_eq = np.ones(len(A_eq))

    # randoms
    random_episodes = []
    for t in range(n_episodes):
        print("Episode", t)
        if t == 0:
            policy_1 = first_policy
        else:
            # epsilon exploration
            if np.random.uniform(0, 1) < 0:  # 0.8**t:
                policy_1 = helper.get_random_policy(mdp)
                print("random")
                random_episodes.append(t)
            else:
                optimization_direction = np.random.uniform(-1, 1, mdp.n_states)
                b_ineq = np.zeros(len(constraint_matrix))
                lp_results = linprog(c=optimization_direction, A_ub=constraint_matrix, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
                reward_function = lp_results.x
                # print("Rewards1:", reward_function[0:49])
                print(reward_function[mdp.reward_states[0]], reward_function[mdp.reward_states[1]], reward_function[mdp.reward_states[2]])
                # compute optimal policy w.r.t. reward_function
                new_mdp = copy.deepcopy(mdp)
                new_mdp.R = reward_function
                joint_policy, V, Q = mdp_solvers.value_iteration_MG(new_mdp)
                policy_1 = extract_policy_one(mdp, joint_policy)
        # get (optimal) response of Agent 2
        policy_2, joint_policy = responses.optimal_response(mdp, policy_1)
        # add constraints according to Theorem 1 / Corollary 1
        cond_MDP = environments.ConditionedMDP(mdp, policy_1)
        new_constraints = get_constraint_matrix(cond_MDP, policy_2)
        constraint_matrix = np.append(constraint_matrix, new_constraints, axis=0)
        print("Size Constraint Matrix:", len(constraint_matrix))
        if mdp.n_states == 392:
            constraint_matrix = get_redundant_constraints(constraint_matrix)
        print("Size Reduced Matrix:", len(constraint_matrix))
        # Evaluate policy_1
        achieved_V, achieved_Q = mdp_solvers.policy_evaluation_MG(mdp, joint_policy)
        payoff.append(achieved_V[mdp.start_state])
        print("Episode", t, ";  Achieved Payoff", achieved_V[mdp.start_state])
    # np.savetxt('episodes_with_random_pol.txt', random_episodes)
    return np.array(payoff)


# The maximum-margin linear programming approach introduced by Ng and Russell (2000)
# TODO: Implement the Algorithm, see https://github.com/aaronsnoswell/irl_methods/blob/master/irl_methods/linear_programming.py
def ng_and_russell(mdp, first_policy):
    # get (optimal) response of Agent 2
    policy_2, joint_policy = responses.optimal_response(mdp, first_policy)
    # calculate constraints according to Theorem 1
    cond_MDP = environments.ConditionedMDP(mdp, first_policy)
    # get policy_2 in a different form:
    vis_policy_2 = np.zeros(cond_MDP.n_states)
    for s in range(cond_MDP.n_states):
        vis_policy_2[s] = np.argmax(policy_2[s])
    # get a sorted transition kernel where the action taken by policy_2 is action 0
    new_transition_kernel = cond_MDP.P.copy()
    for s in range(mdp.n_states):
        new_transition_kernel[s, 0, :] = cond_MDP.P[s, int(vis_policy_2[s]), :]
        new_transition_kernel[s, int(vis_policy_2[s]), :] = cond_MDP.P[s, 0, :]
    cond_MDP.P = new_transition_kernel
    # constraints
    A_ub = np.zeros(shape=[0, mdp.n_states], dtype=float)
    A_ub = get_constraint_matrix(cond_MDP, policy_2)
    # define optimisation direction
    c = np.zeros(shape=[1, mdp.n_states], dtype=float)

    """ Augment objective to add the maximum-margin heuristic. """
    # Expand the c vector add new terms for the min{} operator
    c = np.hstack((c, -1 * np.ones(shape=[1, mdp.n_states])))
    css_offset = c.shape[1] - mdp.n_states
    A_ub = np.hstack((A_ub, np.zeros(shape=[A_ub.shape[0], mdp.n_states])))

    # get occupancy matrix
    occupancy_matrix = np.linalg.inv(np.identity(mdp.n_states) - mdp.gamma * new_transition_kernel[:, 0, :])
    # Add min{} operator constraints
    for i in range(mdp.n_actions_2 - 1):
        # Generate the costly single step constraint terms
        constraint_rows = -1 * (new_transition_kernel[:, 0, :] - new_transition_kernel[:, i, :]) @ occupancy_matrix

        # constraint_rows is nxn - we need to add the min{} terms though
        min_operator_entries = np.identity(mdp.n_states)

        # And we have to make sure we put the min{} operator entries in
        # the correct place in the A_ub matrix
        num_padding_cols = css_offset - mdp.n_states
        padding_entries = np.zeros(shape=[constraint_rows.shape[0], num_padding_cols])
        constraint_rows = np.hstack((constraint_rows, padding_entries, min_operator_entries))

        # Finally, add the new constraints
        A_ub = np.vstack((A_ub, constraint_rows))
        # b_ub = np.vstack((b_ub, np.zeros(shape=[constraint_rows.shape[0], 1])))

    # add constraint r_i <= r_max
    # max_reward = max(mdp.R)
    # for s in range(mdp.n_states):
    #     A = np.zeros(mdp.n_states)
    #     A[s] = 1
    #     A_ub = np.append(A_ub, np.array([A]), axis=0)

    b_ub = np.zeros(len(A_ub))
    lp_results = linprog(c=c, A_ub=A_ub, b_ub=b_ub)

    print("The reward function:", lp_results.x[0:49])

    # Evaluate Reward Function:
    new_mdp = copy.deepcopy(mdp)
    new_mdp.R = lp_results.x
    joint_policy, V, Q = mdp_solvers.value_iteration_MG(new_mdp)
    policy_1 = extract_policy_one(mdp, joint_policy)
    policy_2, joint_policy = responses.optimal_response(mdp, policy_1)
    achieved_V, achieved_Q = mdp_solvers.policy_evaluation_MG(mdp, joint_policy)
    print("Value", achieved_V[mdp.start_state])

    return achieved_V[mdp.start_state]


""" Helping Functions """


def get_constraint_matrix(cond_MDP, policy_2):
    occupancy_matrix = get_occupancy_matrix(cond_MDP, policy_2)
    # (cond_MDP.n_actions_2 - 1) because one of the actions is actually being taken by Agent 2
    constraint_matrix = np.zeros([cond_MDP.n_states * (cond_MDP.n_actions_2 - 1), cond_MDP.n_states])
    # get policy_2 in a different form
    vis_policy_2 = np.zeros(cond_MDP.n_states)
    for s in range(cond_MDP.n_states):
        vis_policy_2[s] = np.argmax(policy_2[s])
    i = 0
    # get an inequality
    for s in range(cond_MDP.n_states):
        for b in range(cond_MDP.n_actions_2):
            if b == vis_policy_2[s]:
                continue
            for next_state in range(cond_MDP.n_states):
                # *-1 because we need < 0 not > 0   # the computation below is a bit of a mindf**k, but actually yields what we want
                constraint_matrix[i] += (cond_MDP.get_transition_probability(s, b, next_state) - cond_MDP.get_transition_probability(s, int(vis_policy_2[s]), next_state)) * occupancy_matrix[next_state]
            i += 1
    return constraint_matrix


# get occupancy matrix w.r.t. policy_1 and policy_2
def get_occupancy_matrix(cond_MDP, policy_2):
    marg_P = cond_MDP.gamma * cond_MDP.get_marginalised_transition_kernel(policy_2)
    matrix = np.identity(cond_MDP.n_states) - marg_P
    return np.linalg.inv(matrix)


def extract_policy_one(mdp, joint_policy):
    policy = np.zeros([mdp.n_states, mdp.n_actions_1])
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions_1):
            policy[s, a] = sum(joint_policy[s, a, :])
    return policy


# simple way of removing redundant constraints, if A[i] <= A[j] element-wise, then A[i] is redundant, since we consider non-negative rewards only.
def get_redundant_constraints(constraint_matrix):
    unique_entries = np.unique(constraint_matrix, axis=0)
    indices = []
    for i, j in itertools.combinations(range(len(unique_entries)), 2):
        if np.all(unique_entries[i] > unique_entries[j]):
            indices.append(j)
        elif np.all(unique_entries[j] > unique_entries[i]):
            indices.append(i)
    reduced_matrix = np.delete(unique_entries, indices, axis=0)
    return reduced_matrix

# # simple way of removing redundant constraints, if A[i] <= A[j] element-wise, then A[i] is redundant, since we consider non-negative rewards only.
# def get_redundant_constraints(constraint_matrix):
#     matrix_size = len(constraint_matrix)
#     indices = []
#     # use itertools here instead
#     for i in range(matrix_size):
#         for j in range(matrix_size):
#             if i != j:
#                 if np.all(constraint_matrix[i] >= constraint_matrix[j]):
#                     indices.append(j)
#     reduced_matrix = np.delete(constraint_matrix, indices, axis=0)
#     return reduced_matrix

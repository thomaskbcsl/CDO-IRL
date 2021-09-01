import numpy as np
import copy
import scipy.stats
import approx_planning
import responses
import mdp_solvers
import environments
import helper

""" Algorithm 2 """

""" TODO: 
    1. Fix reward proposal 
    
    2. It may make more sense to use the mean or the MAP estimate from the samples in episode t-1.
        It seems that whether mean or MAP does not make a huge difference. 
    """


# test run for Partial Information and Boltzmann-rational responses
def test_run_bayes(mdp, true_beta, width, n_episodes, sample_size, first_policy):
    # per episode return
    payoff = [0]

    # sampled rewards and betas
    sampled_rewards = []
    sampled_betas = []

    # list of trajectories
    trajectories = []

    # save record of conditioned MDPs, already computed Q-values w.r.t. reward functions
    cond_MDPs = []
    precomputed_Q = []
    precomputed_R = []

    # initial values for r and beta
    initial_reward = np.zeros(mdp.n_states)
    indices = np.random.choice(mdp.n_states, int(1 / width))
    for i in indices:
        initial_reward[i] += width
    sampled_rewards.append(initial_reward)

    # prior over beta
    sampled_betas.append(prior_beta())

    for t in range(n_episodes):
        if t == 0:
            policy_1 = first_policy
        else:
            new_mdp = copy.deepcopy(mdp)
            # new_mdp.R = map_reward_function
            # policy_1, V, Q = approx_planning.approx_value_iteration_boltzmann(new_mdp, map_beta)
            new_mdp.R = sampled_rewards[-1]
            policy_1 = helper.get_random_policy(mdp)
            # policy_1, V, Q = approx_planning.approx_value_iteration_boltzmann(new_mdp, sampled_betas[-1])

        # get Boltzmann-response of Agent 2
        policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1, true_beta)
        # evaluate policy_1
        achieved_V, achieved_Q = mdp_solvers.policy_evaluation_MG(mdp, joint_policy)
        payoff.append(achieved_V[mdp.start_state])
        print("Episode", t, "; Achieved Payoff", achieved_V[mdp.start_state])

        # Relevant for episode t+1
        print("Episode", t + 1)
        # compute trajectory in episode t
        trajectory = []
        cond_MDP = environments.ConditionedMDP(mdp, policy_1)
        # add cond_MDP to precomputed MDPs
        cond_MDPs.append(cond_MDP)
        i = 0
        while True:
            state_i = cond_MDP.current_state
            action_i = np.random.choice(cond_MDP.n_actions_2, p=policy_2[cond_MDP.current_state, :])
            cond_MDP.current_state = np.random.choice(cond_MDP.n_states, p=cond_MDP.get_transition_probabilities(state_i, action_i))
            trajectory.append([state_i, action_i])
            i += 1
            if i > 30:
                break
            # if np.random.uniform(0, 1) > cond_MDP.gamma:
            #     break
        trajectories.append(trajectory)
        print("Trajectory Length", len(trajectory))

        # sample from posterior
        sampled_rewards, sampled_betas, precomputed_Q, precomputed_R, map_reward_function, map_beta = sample_from_posterior(mdp, width, sample_size, sampled_rewards[-1], sampled_betas[-1], cond_MDPs, precomputed_Q, precomputed_R, trajectories)

    return np.array(payoff)


# Sample from posterior given trajectories
def sample_from_posterior(mdp, width, sample_size, last_episode_last_reward, last_episode_last_beta, cond_MDPs, precomputed_Q, precomputed_R, trajectories):
    # results
    sampled_rewards = []
    sampled_betas = []

    last_reward_function = last_episode_last_reward.copy()
    last_beta = last_episode_last_beta
    old_likelihood, placeholder_Q, placeholder_R = compute_likelihood(last_reward_function, last_beta, cond_MDPs, precomputed_Q, precomputed_R, trajectories)
    for k in range(sample_size):
        # sample r and beta from proposal distribution
        proposed_reward_function, proposed_beta = proposal_distribution(mdp, width, last_reward_function, last_beta)
        pdfs_beta, pdfs_R = get_proposal_pdf(mdp, width, proposed_reward_function, last_reward_function, proposed_beta, last_beta)

        # compute P(tau_1, ..., tau_t | r, beta)
        likelihood, precomputed_Q, precomputed_R = compute_likelihood(proposed_reward_function, proposed_beta, cond_MDPs, precomputed_Q, precomputed_R, trajectories)

        p = likelihood / pdfs_beta[0]
        p_old = old_likelihood / pdfs_beta[1]
        if np.random.uniform(0, 1) < (p / p_old):
            last_reward_function = proposed_reward_function
            last_beta = proposed_beta
            old_likelihood = likelihood
            print("Accept new sample in step", k)
            print("Beta", last_beta)
        sampled_rewards.append(last_reward_function)
        sampled_betas.append(last_beta)
    # print("Episode", len(trajectories))
    print("Last Sample", last_reward_function[mdp.reward_states[0]], last_reward_function[mdp.reward_states[1]], last_reward_function[mdp.reward_states[2]])
    # print(last_reward_function[0:49])
    # print(last_reward_function[49*7:49*8])
    print("Last Beta Sample", last_beta)
    mean_reward = np.zeros(mdp.n_states)
    for reward_function in sampled_rewards:
        mean_reward += reward_function
    mean_reward /= len(sampled_rewards)
    print("MEAN REWARD", mean_reward[0:49])
    print("Mean", mean_reward[mdp.reward_states[0]], mean_reward[mdp.reward_states[1]], mean_reward[mdp.reward_states[2]])
    list_sampled_rewards = [list(item) for item in sampled_rewards]
    map_reward_function = np.array(max(list_sampled_rewards, key=list_sampled_rewards.count))
    print("MAP REWARD", map_reward_function[0:49])
    print("MAP", map_reward_function[mdp.reward_states[0]], map_reward_function[mdp.reward_states[1]], map_reward_function[mdp.reward_states[2]])
    # sampled_rewards.append(map_reward_function)

    map_beta = max(sampled_betas, key=sampled_betas.count)
    print("MAP BETA", map_beta)
    # sampled_betas.append(map_beta)
    return sampled_rewards, sampled_betas, precomputed_Q, precomputed_R, map_reward_function, map_beta


# compute likelihood
def compute_likelihood(reward_function, beta, cond_MDPs, pre_Q, pre_R, trajectories):
    precomputed_Q = pre_Q.copy()
    precomputed_R = pre_R.copy()
    # add empty lookup list for the t-th episode
    precomputed_Q.append([])
    precomputed_R.append([])
    likelihood = 1
    for t in range(len(trajectories)):
        list_precomputed_R = [list(item) for item in precomputed_R[t]]
        # if we have solved the MDP in episode t for this reward function already, re-use the Q-values
        if list(reward_function) in list_precomputed_R:
            index = list_precomputed_R.index(list(reward_function))
            Q_matrix = precomputed_Q[t][index]
            print("re-used")
        else:
            # for the conditioned MDP in episode t, set new reward function
            cond_MDPs[t].R = reward_function
            # solve conditioned MDP to get the Q-values
            policy_2, V, Q_matrix = mdp_solvers.value_iteration(cond_MDPs[t])
            # add reward function and newly calculated Q-values to precomputed list
            precomputed_R[t].append(reward_function)
            precomputed_Q[t].append(Q_matrix)
        # compute likelihood P(tau_t | r, beta)
        Q_exponential = np.exp(beta * Q_matrix)
        for s, b in trajectories[t]:
            likelihood *= 5 * Q_exponential[s, b] / np.sum(Q_exponential[s, :])
    return likelihood, precomputed_Q, precomputed_R


# We use Uniform over neighbours and clipped Normal(mu, 1) for proposal distributions for the reward function and beta, respectively.
def proposal_distribution(mdp, width, last_reward, last_beta):
    proposed_beta = np.random.normal(last_beta, 4)
    proposed_beta = round(proposed_beta, 3)
    if proposed_beta < 0:
        proposed_beta = 0
    # Simplex-Walk
    indices_plus = np.random.choice(np.where(last_reward <= 1 - width)[0], 2)  # add delta to these states
    proposed_reward_function = last_reward.copy()
    for s in indices_plus:
        proposed_reward_function[s] += width
        index_minus = np.random.choice(np.where(proposed_reward_function >= width)[0])
        i = 0
        while s == index_minus:
            index_minus = np.random.choice(np.where(proposed_reward_function >= width)[0])
            i += 1
            if i > 50:
                print("no new found")
                break
        proposed_reward_function[index_minus] -= width
    proposed_reward_function = np.round(proposed_reward_function, 2)  # to get more precise results as python's float numbers can become imprecise over time
    return proposed_reward_function, proposed_beta


# retrieve g(r | r'), g(r' | r), g(beta | beta') and g(beta' | beta)
def get_proposal_pdf(mdp, simplex_width, proposed_R, last_R, proposed_beta, last_beta):
    # for beta: g(proposed | old)
    if proposed_beta == 0:  # we clip the normal distribution, the probability is given by the cdf
        pdf_beta_proposed_given_last = scipy.stats.norm(last_beta, 1).cdf(proposed_beta)
    else:
        pdf_beta_proposed_given_last = scipy.stats.norm(last_beta, 1).pdf(proposed_beta)
    # g(old | proposed)
    if last_beta == 0:
        pdf_beta_last_given_proposed = scipy.stats.norm(proposed_beta, 1).cdf(last_beta)
    else:
        pdf_beta_last_given_proposed = scipy.stats.norm(proposed_beta, 1).pdf(last_beta)

    # the proposal distribution for the reward function is symmetric, for later reference, we return a placeholder
    return [1, 1], [1, 1]   # [pdf_beta_proposed_given_last, pdf_beta_last_given_proposed], [1, 1]


# prior distribution beta
def prior_beta():
    return np.random.exponential(1)


""" Performance when playing the SAME policy over and over again. """


# test run playing the SAME policy over and over again, i.e. single-agent BIRL
def test_run_single(mdp, true_beta, width, n_episodes, sample_size, first_policy):
    # per episode return
    payoff = [0]

    # list of trajectories
    trajectories = []

    # save record of conditioned MDPs, already computed Q-values w.r.t. reward functions
    cond_MDPs = []
    precomputed_Q = []
    precomputed_R = []

    # initial values for r and beta
    last_reward_function = np.zeros(mdp.n_states)
    indices = np.random.choice(mdp.n_states, int(1 / width))
    for i in indices:
        last_reward_function[i] += width

    # fully agnostic prior distribution
    last_beta = 300

    # get Boltzmann-response of Agent 2 FOR LEARNING
    policy_2_IRL, joint_policy_IRL = responses.boltzmann_response(mdp, first_policy, true_beta)

    for t in range(n_episodes):
        if t > 0:
            last_reward_function = sampled_rewards[-1]
            last_beta = sampled_betas[-1]
            new_mdp = copy.deepcopy(mdp)
            new_mdp.R = last_reward_function
            policy_1, V, Q = approx_planning.approx_value_iteration_boltzmann(new_mdp, last_beta)
        # get Boltzmann-response of Agent 2 TO MEASURE PERFORMANCE
        policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1, true_beta)

        # evaluate policy_1
        achieved_V, achieved_Q = mdp_solvers.policy_evaluation_MG(mdp, joint_policy)
        payoff.append(achieved_V[mdp.start_state])
        print("Episode", t, "; Achieved Payoff", achieved_V[mdp.start_state])

        # Relevant for episode t+1
        print("Episode", t + 1)
        # compute trajectory in episode t
        trajectory = []
        cond_MDP = environments.ConditionedMDP(mdp, first_policy)  # use "first_policy" instead of policy_1 !
        # add cond_MDP to precomputed MDPs
        cond_MDPs.append(cond_MDP)
        i = 0
        while True:
            state_i = cond_MDP.current_state
            action_i = np.random.choice(cond_MDP.n_actions_2, p=policy_2_IRL[cond_MDP.current_state, :])  # important to use policy_2_IRL here. We play the same policy over and over again
            cond_MDP.current_state = np.random.choice(cond_MDP.n_states, p=cond_MDP.get_transition_probabilities(state_i, action_i))
            trajectory.append([state_i, action_i])
            i += 1
            if i > 30:
                break
            # if np.random.uniform(0, 1) > cond_MDP.gamma:
            #     break
        trajectories.append(trajectory)
        print("Trajectory Length", len(trajectory))

        # sample from posterior
        sampled_rewards, sampled_betas, precomputed_Q, precomputed_R = sample_from_posterior(mdp, width, sample_size, last_reward_function, last_beta, cond_MDPs, precomputed_Q, precomputed_R, trajectories)
    return np.array(payoff)

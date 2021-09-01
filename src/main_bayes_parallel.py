import numpy as np
import copy
import scipy.stats
from multiprocessing import Pool
import approx_planning
import responses
import mdp_solvers
import environments
import helper

""" Code to Run the Experiments """

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
    # initial_reward = np.zeros(mdp.n_states)
    # indices = np.random.choice(mdp.n_states, int(1 / width))
    # for i in indices:
    #     initial_reward[i] += width
    sampled_rewards.append(np.random.dirichlet(np.ones(mdp.n_states)))

    # prior over beta
    sampled_betas.append(prior_beta())
    c, d = proposal_distribution(mdp, width, sampled_rewards[-1], sampled_betas[-1])
    A = get_proposal_pdf(mdp, width, sampled_rewards[-1], sampled_betas[-1], c, d)

    for t in range(n_episodes):
        if t == 0:
            policy_1 = first_policy
        else:
            new_mdp = copy.deepcopy(mdp)
            # new_mdp.R = map_reward_function
            # policy_1, V, Q = approx_planning.approx_value_iteration_boltzmann(new_mdp, map_beta)
            new_mdp.R = sampled_rewards[-1]
            # policy_1 = helper.get_random_policy(mdp)
            policy_1, V, Q = approx_planning.approx_value_iteration_boltzmann(new_mdp, sampled_betas[-1])

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
    old_likelihood, placeholder_Q, placeholder_R = compute_likelihood_parallel(last_reward_function, last_beta, cond_MDPs, precomputed_Q, precomputed_R, trajectories)
    for k in range(sample_size):
        # sample r and beta from proposal distribution
        proposed_reward_function, proposed_beta = proposal_distribution(mdp, width, last_reward_function, last_beta)
        pdfs_beta, pdfs_R = get_proposal_pdf(mdp, width, proposed_reward_function, last_reward_function, proposed_beta, last_beta)

        # compute P(tau_1, ..., tau_t | r, beta)
        likelihood, precomputed_Q, precomputed_R = compute_likelihood_parallel(proposed_reward_function, proposed_beta, cond_MDPs, precomputed_Q, precomputed_R, trajectories)

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


# compute likelihood MULTIPROCESSING
def compute_likelihood_parallel(reward_function, beta, cond_MDPs, pre_Q, pre_R, trajectories):
    precomputed_Q = pre_Q.copy()
    precomputed_R = pre_R.copy()
    # add empty lookup list for the t-th episode
    precomputed_Q.append([])
    precomputed_R.append([])
    likelihood = 1
    # print(__name__)
    for t in range(len(trajectories)):
        cond_MDPs[t].R = reward_function
    if __name__ == '__main__':
        pool = Pool()
        Q_matrices = pool.map(task, cond_MDPs)
        pool.close()
        pool.join()
        for t in range(len(trajectories)):
            Q_exponential = np.exp(beta * Q_matrices[t])
            for s, b in trajectories[t]:
                likelihood *= 5 * Q_exponential[s, b] / np.sum(Q_exponential[s, :])
        return likelihood, precomputed_Q, precomputed_R


def task(cond_MDP):
    policy_2, V, Q_matrix = mdp_solvers.value_iteration(cond_MDP)
    return Q_matrix


# We use Uniform over neighbours and clipped Normal(mu, 1) for proposal distributions for the reward function and beta, respectively.
def proposal_distribution(mdp, width, last_reward, last_beta):
    # clipped normal
    proposed_beta = np.random.normal(last_beta, 2)
    proposed_beta = round(proposed_beta, 3)
    if proposed_beta < 0:
        proposed_beta = 0

    # Dirichlet proposal
    proposed_reward_function = np.random.dirichlet(last_reward + 0.5)
    proposed_reward_function = np.round(proposed_reward_function, 5)
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

    # for R
    A = last_R
    B = last_R + 0.5
    pdf_reward_proposed_given_last = scipy.stats.dirichlet(last_R+0.5).pdf(proposed_R)
    pdf_reward_last_given_proposed = scipy.stats.dirichlet(proposed_R+0.5).pdf(last_R)
    return [pdf_beta_proposed_given_last, pdf_beta_last_given_proposed], [pdf_reward_proposed_given_last, pdf_reward_last_given_proposed]


# prior distribution beta
def prior_beta():
    return np.random.exponential(1)


mdp = environments.MazeMaker()
mdp.R = mdp.R / sum(mdp.R)
beta = 125
T = 25
width = 1 / 100  # width of the discrete simplex (width-skeleton)
n_iterations = 1
sample_size = 500  # no. of samples from posterior

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
payoff_optimal_joint = np.full(T + 1, opt_V[mdp.start_state])
payoff_optimal_vi = np.full(T + 1, vi_V[mdp.start_state])

payoff_bayes = np.zeros(T + 1)
for i in range(n_iterations):
    first_policy = helper.get_random_policy(mdp)
    payoff_bayes += test_run_bayes(mdp, beta, width, T, sample_size, first_policy)

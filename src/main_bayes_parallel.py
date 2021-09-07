import numpy as np
import copy
import scipy.stats
import matplotlib.pyplot as plt
from multiprocessing import Pool
import approx_planning
import responses
import mdp_solvers
import environments
import helper

""" Code to Run the Experiments """

""" Context-Dependent Online Bayesian IRL via MCMC sampling. """


# test run for Partial Information and Boltzmann-rational responses
def test_run_bayes(mdp, true_beta, n_episodes, sample_size, first_policy):
    # per episode return
    payoff = [0]

    # sampled rewards and betas
    sampled_rewards = []
    sampled_betas = []

    # list of trajectories
    trajectories = []

    # save record of conditioned MDPs
    cond_MDPs = []

    # prior over rewards
    sampled_rewards.append(np.random.dirichlet(np.ones(mdp.n_states)))

    # prior over beta
    sampled_betas.append(prior_beta(mdp, true_beta))
    # c, d = proposal_distribution(mdp, sampled_rewards[-1], sampled_betas[-1])
    # A = get_proposal_pdf(mdp, sampled_rewards[-1], c, sampled_betas[-1], d)

    for t in range(n_episodes):
        if t == 0:
            policy_1 = first_policy
        else:
            mean_reward = np.zeros(mdp.n_states)
            for reward_function in sampled_rewards:
                mean_reward += reward_function
            mean_reward /= len(sampled_rewards)
            new_mdp = copy.deepcopy(mdp)
            new_mdp.R = mean_reward
            mean_beta = 0
            for beta in sampled_betas:
                mean_beta += beta
            mean_beta /= len(sampled_betas)
            # sampled_rewards.append(mean_reward)
            # sampled_betas.append(mean_beta)
            print("Mean Reward", mean_reward[0:49])
            print("Mean Beta", mean_beta)
            policy_1, V, Q = approx_planning.approx_value_iteration_boltzmann(new_mdp, mean_beta)

        # get Boltzmann-response of Agent 2
        policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1, true_beta)
        # evaluate policy_1
        achieved_V, achieved_Q = mdp_solvers.policy_evaluation_MG(mdp, joint_policy)
        payoff.append(achieved_V[mdp.start_state])
        print("Episode", t, "; Achieved Payoff", achieved_V[mdp.start_state])

        # Relevant for episode t+1
        print("Episode", t + 1)

        ########################################################### FOR TESTING ONLY - REMOVE LATER
        # policy_1 = helper.get_random_policy(mdp)
        # policy_2, joint_policy = responses.boltzmann_response(mdp, policy_1, true_beta)
        ############################################################
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
            if i > 2 and np.random.uniform(0, 1) > cond_MDP.gamma:
                break
        trajectories.append(trajectory)
        print("Trajectory Length", len(trajectory))

        if t == n_episodes - 1:
            break
        # sample from posterior
        sample_size_now = sample_size   # int(((t+1)**1.3)*100)    # int(sample_size/(n_episodes - t))     # int((n_episodes+1)*100)
        print("sample size", sample_size_now)
        sampled_rewards, sampled_betas = sample_from_posterior(mdp, sample_size_now, sampled_rewards[-1], sampled_betas[-1], cond_MDPs, trajectories)
    return np.array(payoff)


# Sample from posterior given trajectories
def sample_from_posterior(mdp, sample_size, last_episode_last_reward, last_episode_last_beta, cond_MDPs, trajectories):
    # results
    sampled_rewards = []
    sampled_betas = []

    # last_reward_function, last_beta = proposal_distribution(mdp, last_episode_last_reward, last_episode_last_beta)
    last_reward_function = last_episode_last_reward.copy()
    last_beta = last_episode_last_beta
    old_likelihood = compute_likelihood_parallel(last_reward_function, last_beta, cond_MDPs, trajectories)
    for k in range(sample_size):
        # sample r and beta from proposal distribution
        proposed_reward_function, proposed_beta = proposal_distribution(mdp, last_reward_function, last_beta)
        pdfs_beta, pdfs_R = get_proposal_pdf(mdp, proposed_reward_function, last_reward_function, proposed_beta, last_beta)

        # compute P(tau_1, ..., tau_t | r, beta)
        likelihood = compute_likelihood_parallel(proposed_reward_function, proposed_beta, cond_MDPs, trajectories)

        # p = likelihood / pdfs_beta[0]
        # p_old = old_likelihood / pdfs_beta[1]
        p = likelihood
        p_old = old_likelihood

        if np.random.uniform(0, 1) < (p / p_old):
            last_reward_function = proposed_reward_function
            last_beta = proposed_beta
            old_likelihood = likelihood
            print("Accept new sample in step", k)
            print("Beta", last_beta)
        sampled_rewards.append(last_reward_function)
        sampled_betas.append(last_beta)
    return sampled_rewards, sampled_betas


# compute likelihood MULTIPROCESSING
def compute_likelihood_parallel(reward_function, beta, cond_MDPs, trajectories):
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
        return likelihood


def task(cond_MDP):
    policy_2, V, Q_matrix = mdp_solvers.value_iteration(cond_MDP)
    return Q_matrix


# We use Uniform over neighbours and clipped Normal(mu, 1) for proposal distributions for the reward function and beta, respectively.
def proposal_distribution(mdp, last_reward, last_beta):
    # gamma proposal
    proposed_beta = np.random.gamma(last_beta, 1 + 5 / last_beta)
    proposed_beta = round(proposed_beta, 1)

    proposed_beta = last_beta

    # Dirichlet proposal
    # proposed_reward_function = np.random.dirichlet(last_reward + 0.5)
    proposed_reward_function = np.random.dirichlet(np.ones(mdp.n_states))
    proposed_reward_function = np.round(proposed_reward_function, 5)
    return proposed_reward_function, proposed_beta


# retrieve g(r | r'), g(r' | r), g(beta | beta') and g(beta' | beta)
def get_proposal_pdf(mdp, proposed_R, last_R, proposed_beta, last_beta):
    # for beta: g(proposed | old)
    # eps = 0.5  # add epsilon and compute cdfs as we otherwise face division by zero
    pdf_beta_proposed_given_last = scipy.stats.gamma(last_beta, 1 + 5 / last_beta).pdf(proposed_beta)  # - scipy.stats.gamma(last_beta, 1).cdf(proposed_beta - eps)
    # g(old | proposed)
    pdf_beta_last_given_proposed = scipy.stats.gamma(proposed_beta, 1 + 5 / last_beta).pdf(last_beta)  # - scipy.stats.gamma(proposed_beta, 1).cdf(last_beta - eps)

    # for R
    # pdf_reward_proposed_given_last = scipy.stats.dirichlet.pdf(proposed_R, last_R+0.5)    # this has FloatingPointErrors for large state spaces.
    # pdf_reward_last_given_proposed = scipy.stats.dirichlet.pdf(last_R, proposed_R+0.5)
    return [1, 1], [1, 1] # [pdf_beta_proposed_given_last + 1e-10, pdf_beta_last_given_proposed + 1e-10], [1, 1]

# prior distribution beta
def prior_beta(mdp, beta):
    return round(beta * sum(mdp.R), 2)    #np.random.gamma(10, 1)  # 480


# pdf of prior
def pdf_prior_beta(beta):
    return scipy.stats.gamma(5, 1).pdf(beta)


mdp = environments.MazeMaker()    ## beta = 20
beta = 10   # 480
# mdp = environments.RandomMDP(n_states=200, n_actions_1=4, n_actions_2=4)  ## beta = 10
# beta = 10  # approx 1000
mdp.R = mdp.R  # / sum(mdp.R)
T = 40
n_iterations = 1
sample_size = 10000  # no. of samples from posterior

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
joint_V, vi_V = mdp_solvers.policy_comparison_MG(mdp, response_opt_joint_joint, response_vi_policy_joint)
print("Optimal Payoff Joint-Opt", joint_V[mdp.start_state])
print("Optimal Payoff Approx VI", vi_V[mdp.start_state])
# results
payoff_optimal_joint = np.full(T + 1, joint_V[mdp.start_state])
payoff_optimal_vi = np.full(T + 1, vi_V[mdp.start_state])

payoff_bayes = np.zeros(T + 1)
for i in range(n_iterations):
    first_policy = helper.get_random_policy(mdp)
    payoff_bayes = test_run_bayes(mdp, beta, T, sample_size, first_policy)
    regret_vi = np.zeros(T + 1)  # per-episode regret with respect to oracle vi
    regret_joint = np.zeros(T + 1)  # per-episode regret with respect to oracle joint
    for t in range(1, T + 1):
        regret_vi[t] = vi_V[mdp.start_state] - payoff_bayes[t]
        regret_joint[t] = joint_V[mdp.start_state] - payoff_bayes[t]
    cum_regret_vi = np.zeros(T + 1)  # cumulative regret w.r.t. oracle value iteration
    cum_regret_joint = np.zeros(T + 1)  # cumulative regret w.r.t. oracle joint optimal policy
    for t in range(1, T + 1):
        cum_regret_vi[t] = sum(regret_vi[0:t + 1])
        cum_regret_joint[t] = sum(regret_joint[0:t + 1])
    save = np.array([payoff_bayes, regret_vi, regret_joint, cum_regret_vi, cum_regret_joint])
    np.savetxt(str(i) + 'regret_bayes_3.txt', save)
    plt.style.use('ggplot')
    plt.plot(cum_regret_vi, label="Context-Dependent Online Bayesian IRL")
    plt.plot(cum_regret_joint, label="w.r.t. joint")
    plt.legend()
    plt.xlabel("Episode")
    plt.savefig(str(i) + "Partial-Info-Boltz-Dem_3.pdf")
    np.savetxt(str(i) + 'first_policy.txt', first_policy)   # to get a fair comparison

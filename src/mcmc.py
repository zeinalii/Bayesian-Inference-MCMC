import numpy as np
import scipy.stats as st

def log_posterior(X, theta, prior_mean=0, prior_std=1):
    """Calculate the log of the unnormalized posterior."""
    log_likelihood = np.sum(st.norm(loc=theta, scale=1).logpdf(X))
    log_prior = st.norm(loc=prior_mean, scale=prior_std).logpdf(theta)
    return log_likelihood + log_prior

def proposal_distribution(theta_current, proposal_std=0.2):
    """Generate a new proposal based on the current theta."""
    return st.norm(loc=theta_current, scale=proposal_std).rvs()

def metropolis_hastings(X, theta_init, n_iterations=10000, proposal_std=0.2, burn_in=1000):
    """
    Implement the Metropolis-Hastings algorithm for Bayesian inference.
    
    Args:
    X: Observed data
    theta_init: Initial parameter value
    n_iterations: Number of MCMC iterations
    proposal_std: Standard deviation for the proposal distribution
    burn_in: Number of initial samples to discard
    
    Returns:
    thetas: Array of sampled parameter values
    acceptance_rate: Fraction of accepted proposals
    """
    thetas = np.zeros(n_iterations)
    theta_current = theta_init
    accepted = 0
    
    for i in range(n_iterations):
        theta_proposal = proposal_distribution(theta_current, proposal_std)
        
        log_ratio = log_posterior(X, theta_proposal) - log_posterior(X, theta_current)
        acceptance_prob = min(1, np.exp(log_ratio))
        
        if np.random.random() < acceptance_prob:
            theta_current = theta_proposal
            accepted += 1
        
        thetas[i] = theta_current
    
    return thetas[burn_in:], accepted / n_iterations

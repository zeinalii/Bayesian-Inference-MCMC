import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

def generate_synthetic_data(n_samples, true_mean=3, true_std=1):
    """Generate synthetic data from a normal distribution."""
    return np.random.normal(loc=true_mean, scale=true_std, size=n_samples)

def plot_mcmc_results(samples, burn_in, param_name="θ"):
    """Plot MCMC results including trace, distribution, and autocorrelation."""
    samples_post_burnin = samples[burn_in:]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Trace plot (full)
    axes[0, 0].plot(samples)
    axes[0, 0].set_title('Trace (Full)')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel(param_name)
    
    # Trace plot (post burn-in)
    axes[0, 1].plot(samples_post_burnin)
    axes[0, 1].set_title('Trace (Post Burn-in)')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel(param_name)
    
    # Distribution plot
    sns.histplot(samples_post_burnin, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Posterior Distribution')
    axes[1, 0].set_xlabel(param_name)
    
    # Autocorrelation plot
    plot_acf(samples_post_burnin, ax=axes[1, 1], lags=100)
    axes[1, 1].set_title('Autocorrelation')
    
    plt.tight_layout()
    return fig

def calculate_effective_sample_size(samples):
    """Calculate the effective sample size of MCMC samples."""
    n = len(samples)
    acf = np.correlate(samples - np.mean(samples), samples - np.mean(samples), mode='full')
    acf = acf[n-1:] / (n * np.var(samples))
    
    # Find the first negative autocorrelation
    negative_idx = np.where(acf < 0)[0]
    if len(negative_idx) > 0:
        cutoff = negative_idx[0]
    else:
        cutoff = len(acf)
    
    tau = 1 + 2 * np.sum(acf[1:cutoff])
    return int(n / tau)
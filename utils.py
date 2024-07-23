import torch
from torch import Tensor as _T

from scipy.ndimage import gaussian_filter1d

from tqdm import tqdm


def dx_sampler(v: _T, sigma: _T, dt: _T):
    "All args of shape [num_trials]"
    drift_term = v * dt
    diffusion_term = sigma * torch.sqrt(dt) * torch.randn_like(sigma)
    return drift_term + diffusion_term


def check_against_binary_bounds(state: _T, upper_bound: _T, lower_bound: _T):
    "Generate a mask of which trials still need updating. All args of shape [num_trials]"
    below_upper_bound = (state < upper_bound)
    above_lower_bound = (state > lower_bound)
    return (
        torch.logical_and(below_upper_bound, above_lower_bound),
        ~below_upper_bound,
        ~above_lower_bound,
    )

def update_bounds(upper_bound, lower_bound):
    "All args of shape [num_trials]"
    return upper_bound, lower_bound

def recitfy_proposal_against_bounds(proposed_state, upper_bound, lower_bound):
    "All args of shape [num_trials]"
    upper_rectified = torch.min(proposed_state, upper_bound)
    lower_rectified = torch.max(upper_rectified, lower_bound)
    return lower_rectified


def simplest_simulation(
    dt: float,
    a: float,
    w: float,
    nu: float,
    sigma: float,
    num_trials: int,
    duration: float,
):
    """
    Most basic simulation
    All trials are identical, bounds are stationary, time is not warped
        No need to return things like time axis because of this
    Everything can be fed in as a float!
    Assume that upper bound is correct answer
    """
    assert nu >= 0.0, 'Drift strength should be non-negative'

    initial_conditions = torch.ones(num_trials) * w * a / 2

    # Convert parameters to correct form for loop
    num_time_steps = int(duration // dt + 1)
    nu = nu * torch.ones(num_trials)
    sigma = sigma * torch.ones(num_trials)
    dt = dt * torch.ones(num_trials)
    upper_bound = + a / 2 * torch.ones(num_trials)
    lower_bound = - a / 2 * torch.ones(num_trials)

    # Initial iteration over time
    all_states = [initial_conditions]
    num_hit_upper = []
    num_hit_lower = []
    
    # Iteration over time
    for _ in tqdm(range(num_time_steps)):

        # Current state
        x = all_states[-1].clone()

        # Check which trials still have not hit bound
        undecided_mask, hit_upper_mask, hit_lower_mask = check_against_binary_bounds(
            x, upper_bound, lower_bound
        )

        # Update bounds
        upper_bound, lower_bound = update_bounds(upper_bound, lower_bound)
        
        # For those trials, calculate the stochastic step and propose the addition, then rectify it
        dx = dx_sampler(nu[undecided_mask], sigma[undecided_mask], dt[undecided_mask])
        proposal = x[undecided_mask] + dx
        rectified_proposal = recitfy_proposal_against_bounds(proposal, upper_bound[undecided_mask], lower_bound[undecided_mask])

        # Update most recent state and save
        x[undecided_mask] = rectified_proposal
        all_states.append(x)

        # Track how many have hit upper/lower
        num_hit_upper.append(hit_upper_mask.sum().item())
        num_hit_lower.append(hit_lower_mask.sum().item())

    # Join all trajectories
    trajectory = torch.stack(all_states, 1)
    num_hit_upper = torch.tensor(num_hit_upper)
    num_hit_lower = torch.tensor(num_hit_lower)

    return trajectory, num_hit_upper, num_hit_lower, hit_upper_mask, hit_lower_mask
    

def generate_RT_distributions_no_time_warp(num_hit_upper, num_hit_lower, dt):
    "Both of shape [timesteps], assume constant dt for now!"
    total_hit_upper = num_hit_upper[-1]
    total_hit_lower = num_hit_lower[-1]
    num_hit_upper_cdf = num_hit_upper / total_hit_upper
    num_hit_lower_cdf = num_hit_lower / total_hit_lower
    num_hit_upper_pdf = num_hit_upper.diff() / dt / total_hit_upper
    num_hit_lower_pdf = num_hit_lower.diff() / dt / total_hit_lower
    smoothed_num_hit_upper_pdf = gaussian_filter1d(num_hit_upper_pdf.numpy(), 10.0)
    smoothed_num_hit_lower_pdf = gaussian_filter1d(num_hit_lower_pdf.numpy(), 10.0)
    return (
        total_hit_upper,
        total_hit_lower,
        num_hit_upper_cdf,
        num_hit_lower_cdf,
        num_hit_upper_pdf,
        num_hit_lower_pdf,
        smoothed_num_hit_upper_pdf,
        smoothed_num_hit_lower_pdf
    )

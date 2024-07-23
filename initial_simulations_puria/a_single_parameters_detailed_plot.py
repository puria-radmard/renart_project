from matplotlib import pyplot as plt
from utils import *

# Simulation parameters
dt = 0.001
a = 1.0
w = 0.0
nu = 5.0
sigma = 1.0
num_trials = 100_000
duration = 1.0

trajectory, num_hit_upper, num_hit_lower, hit_upper_mask, hit_lower_mask = simplest_simulation(
    dt, a, w, nu, sigma, num_trials, duration
)

(
    total_hit_upper,
    total_hit_lower,
    num_hit_upper_cdf,
    num_hit_lower_cdf,
    num_hit_upper_pdf,
    num_hit_lower_pdf,
    smoothed_num_hit_upper_pdf,
    smoothed_num_hit_lower_pdf
) = generate_RT_distributions_no_time_warp(num_hit_upper, num_hit_lower, dt)

# Get average, seperated by answer
trajectories_hit_upper = trajectory[hit_upper_mask].mean(0).numpy()
trajectories_hit_lower = trajectory[hit_lower_mask].mean(0).numpy()

#### Plot master plot
fig, axes = plt.subplot_mosaic(
    '''
    NNNNNN
    NNNNNN
    BBBBBB
    BBBBBB
    AAAAAA
    AAAAAA
    AAAAAA
    AAAAAA
    AAAAAA
    AAAAAA
    CCCCCC
    CCCCCC
    VVVVVV
    VVVVVV
    ''',
    figsize = (10, 5)
)


time_axis = (torch.ones_like(trajectory[0]) * dt).cumsum(0)

for traj in trajectory[:50]:
    color = 'blue' if traj[-1] > 0 else 'red'
    axes['A'].plot(time_axis, traj.numpy(), color = color, alpha = 0.05)

axes['A'].plot(time_axis, trajectories_hit_upper, color = 'blue')
axes['A'].plot(time_axis, trajectories_hit_lower, color = 'red')

axes['B'].plot(time_axis[1:-1], num_hit_upper_pdf.numpy())
axes['C'].plot(time_axis[1:-1], - num_hit_lower_pdf.numpy())
axes['B'].plot(time_axis[1:-1], smoothed_num_hit_upper_pdf)
axes['C'].plot(time_axis[1:-1], - smoothed_num_hit_lower_pdf)
axes['N'].plot(time_axis[1:], num_hit_upper_cdf.numpy())
axes['V'].plot(time_axis[1:], - num_hit_lower_cdf.numpy())

for x in 'ABCNV':
    axes[x].set_xlim(axes[x].get_xlim())
    if x != 'V':
        axes[x].get_xaxis().set_visible(False)

axes['A'].set_ylabel('Decision\nvariable')
axes['B'].set_ylabel('RT PDF\ncorrect')
axes['C'].set_ylabel('RT PDF\nincorrect')
axes['N'].set_ylabel('RT CDF\ncorrect')
axes['V'].set_ylabel('RT CDF\nincorrect')

fig.suptitle(f"$a = {a}, w = {w}, \\nu = {nu}, \sigma = {sigma}$")
plt.savefig('initial_simulations_puria/a_single_parameters_detailed_plot.png')

from matplotlib import pyplot as plt
from utils import *

nus = torch.linspace(0.0, 7.0, 10)

dt = 0.001
a = 2.0
w = 0.0
sigma = 1.0
num_trials = 100_000
duration = 1.0

all_mean_reaction_times_correct = []
all_mean_reaction_times_incorrect = []
all_lower_quartile_reaction_times_correct = []
all_lower_quartile_reaction_times_incorrect = []
all_upper_quartile_reaction_times_correct = []
all_upper_quartile_reaction_times_incorrect = []
accuracies = []
all_chose_incorrect = []
all_no_choice = []

fig, axes = plt.subplot_mosaic(
    '''
    AABBCC
    AABBCC
    DDDEEE
    DDDEEE
    ''',
    figsize = (10, 10)
)


for nu in nus:

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

    # Accuracies and choice counts
    all_chose_incorrect.append((num_hit_lower[-1]) / num_trials)
    all_no_choice.append((num_trials - num_hit_upper[-1] - num_hit_lower[-1]) / num_trials)
    accuracies.append(num_hit_upper[-1] / num_trials)

    assert all_chose_incorrect[-1] + all_no_choice[-1] + accuracies[-1] == 1.0

    # E[RT | in/correct], by integrating cdf.diff
    time_axis = (torch.ones_like(num_hit_upper_pdf) * dt).cumsum(0) # no warping of time!
    integral_pt_dt_correct = (num_hit_upper_pdf * dt).cumsum(0)
    integral_pt_dt_incorrect = (num_hit_lower_pdf * dt).cumsum(0)

    all_lower_quartile_reaction_times_correct.append(time_axis[(integral_pt_dt_correct - 0.25).abs().argmin()])
    all_lower_quartile_reaction_times_incorrect.append(time_axis[(integral_pt_dt_incorrect - 0.25).abs().argmin()])
    all_upper_quartile_reaction_times_correct.append(time_axis[(integral_pt_dt_correct - 0.75).abs().argmin()])
    all_upper_quartile_reaction_times_incorrect.append(time_axis[(integral_pt_dt_incorrect - 0.75).abs().argmin()])

    all_mean_reaction_times_correct.append((time_axis * num_hit_upper_pdf * dt).sum())
    all_mean_reaction_times_incorrect.append((time_axis * num_hit_lower_pdf * dt).sum())

    pdf_alpha = ((nu + nus[-1]) / (2 * nus[-1])).item()
    axes['D'].plot(time_axis, smoothed_num_hit_upper_pdf, color = 'blue', alpha = pdf_alpha, label = f"{nu}, {round(accuracies[-1].item(), 2)}")
    axes['E'].plot(time_axis, smoothed_num_hit_lower_pdf, color = 'red', alpha = pdf_alpha, label = f"{nu}, {round(accuracies[-1].item(), 2)}")


axes['D'].set_title('$p(RT | ... , resp. = correct)$')
axes['E'].set_title('$p(RT | ... , resp. = incorrect)$')
axes['E'].legend(title = '$\\nu, acc.$')


axes['A'].plot(nus, all_mean_reaction_times_correct, color = 'blue', label = 'Mean, correct')
axes['A'].fill_between(nus, all_lower_quartile_reaction_times_correct, all_upper_quartile_reaction_times_correct, color = 'blue', alpha = 0.1)
axes['A'].plot(nus, all_mean_reaction_times_incorrect, color = 'red', label = 'Mean, incorrect')
axes['A'].fill_between(nus, all_lower_quartile_reaction_times_incorrect, all_upper_quartile_reaction_times_incorrect, color = 'red', alpha = 0.1)
axes['A'].set_xlabel('$\\nu$')
axes['A'].set_title('$p(RT | \\nu, resp.)$')

axes['B'].plot(accuracies, all_mean_reaction_times_correct, color = 'blue', label = 'Mean, correct')
axes['B'].fill_between(accuracies, all_lower_quartile_reaction_times_correct, all_upper_quartile_reaction_times_correct, color = 'blue', alpha = 0.1)
axes['B'].plot(accuracies, all_mean_reaction_times_incorrect, color = 'red', label = 'Mean, incorrect')
axes['B'].fill_between(accuracies, all_lower_quartile_reaction_times_incorrect, all_upper_quartile_reaction_times_incorrect, color = 'red', alpha = 0.1)
axes['B'].set_xlabel('$acc.$')
axes['B'].set_title('$p(RT | acc., resp.)$')

axes['C'].plot(nus, accuracies, label = 'Correct')
axes['C'].plot(nus, all_chose_incorrect, label = 'Incorrect')
axes['C'].plot(nus, all_no_choice, label = 'No choice in 1.0s')
axes['C'].set_xlabel('$\\nu$')
axes['C'].set_title('$p(resp. | \\nu)$')

axes['A'].legend()
axes['B'].legend()
axes['C'].legend()

plt.savefig('initial_simulations_puria/b_reaction_dependence_on_drift.png')



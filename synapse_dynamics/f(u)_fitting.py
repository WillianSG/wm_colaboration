from lmfit import Minimizer, Parameters, report_fit
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

x = np.array([0, 0.2, 0.7, 0.9, 1])
y = np.array([0, 0, 0.7, 0.9, 1])

plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['figure.titlesize'] = 20
fig = plt.figure(figsize=(25, 25))
gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.5, 1])
ax11 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1], projection='3d')
ax2 = fig.add_subplot(gs[1, :])

# Create object for parameter storing
params_gompertz = Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params_gompertz.add_many(
    ('a', 1, False, None, None, None, None),
    ('b', 3, True, None, None, None, None),
    ('c', 7, True, None, None, None, None)
)

# Define the parameter object
params_genlogistic = Parameters()
# Add parameters and initial values
params_genlogistic.add_many(
    ('A', 0, False, None, None, None, None),
    ('K', 1, False, None, None, None, None),
    ('B', 5, True, None, None, None, None),
    ('Q', 0.01, True, None, None, None, None),
    ('mu', 0.1, True, None, None, None, None),
    ('T', 0.8, True, None, None, None, None)
)


def ca_over_cb(ca, cb):
    ca[3] = 0.5
    cb[3] = 0.5
    out = np.zeros(4)

    out[3] = ca[3] + cb[3] * (1 - ca[3])
    for i in range(0, 3):
        out[i] = (ca[i] * ca[3] + cb[i] * cb[3] * (1 - ca[3])) / out[3]

    out[3] = 1

    return out


def genlogistic(params, x):
    if isinstance(params, Parameters):
        params = params.valuesdict()

    return params['A'] + (params['K'] - params['A']) / (1 + params['Q'] * np.exp(-params['B'] * (x - params['T']))) ** (
            1 / params['mu'])


def genlogistic_derivative(params, x):
    if isinstance(params, Parameters):
        params = params.valuesdict()

    return (params['B'] * params['Q'] * (params['K'] - params['A']) * np.exp(-params['B'] * (x - params['T'])) * (
            params['Q'] * np.exp(-params['B'] * (x - params['T'])) + 1) ** (-1 / params['mu'] - 1)) / params['mu']


def gompertz(params, x):
    if isinstance(params, Parameters):
        params = params.valuesdict()

    return params['a'] * np.exp(-np.exp(params['b'] - params['c'] * x))


def gompertz_derivative(params, x):
    if isinstance(params, Parameters):
        params = params.valuesdict()

    return params['a'] * params['c'] * np.exp(params['b'] - np.exp(params['b'] - params['c'] * x) - params['c'] * x)


def penalty(model, func, x):
    from sklearn import preprocessing

    # calculate distance from f(x)=x as penalty term
    penalty1 = np.abs(model - x)
    # calculate distance from f'(x)=1 as penalty term
    penalty2 = np.abs(eval(func.__name__ + '_derivative')(params, x) - 1)

    penalty1_norm = preprocessing.minmax_scale(penalty1)
    penalty2_norm = preprocessing.minmax_scale(penalty2)

    return (penalty1_norm, penalty2_norm)


def error(params, func, x):
    from sklearn import preprocessing

    model = func(params, x)

    penalty1, penalty2 = penalty(model, func, x)

    penalty1_norm = preprocessing.minmax_scale(penalty1)
    penalty2_norm = preprocessing.minmax_scale(penalty2)

    score = np.mean(penalty1_norm + penalty2_norm)

    return score


# Write down the objective function that we want to minimize, i.e., the residuals
def residuals(params, func, x, data, lam1=0, lam2=0, exclude=[0, 1]):
    model = func(params, x)

    penalty1, penalty2 = penalty(model, func, x)

    # exclude first points from being penalised
    penalty1[exclude] = 0
    penalty2[exclude] = 0

    # Return residuals
    return model - data + lam1 * penalty1 + lam2 * penalty2


lambdas = 10
scores = {}
X = np.linspace(0, 1, lambdas)
Y = np.linspace(0, 1, lambdas)
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)
colors1 = plt.cm.autumn_r(np.linspace(0, 1, lambdas))
colors2 = plt.cm.summer_r(np.linspace(0, 1, lambdas))

functions = [(gompertz, params_gompertz)]
points_to_exclude = [0, 1, 2]

for f, params in functions:
    for i, lam1 in enumerate(X[0]):
        for j, lam2 in enumerate(Y[:, 0]):
            print(f'Fitting {f.__name__} with lambda1 = {lam1}, lambda2 = {lam2}')
            try:
                # Create a Minimizer object
                minner = Minimizer(residuals, params_gompertz, fcn_args=(f, x, y, lam1, lam2, points_to_exclude))
                # Perform the minimization
                fit = minner.minimize()
                # Summarize results
                report_fit(fit)

                # Plot fitted curve
                result = f(fit.params, x)
                # ax11.plot(x, result_gompertz, color=colors[i], markersize=15, label='Gompertz')
                # Get a smooth curve by plugging a time vector to the fitted logistic model
                smooth_x_vec = np.linspace(0, 1, 1000)
                smooth_y_vec = f(fit.params, smooth_x_vec)
                ax2.plot(smooth_x_vec, smooth_y_vec, color=ca_over_cb(colors1[i], colors2[j]), linestyle='--',
                         linewidth=1,
                         alpha=0.4,
                         label=rf'$\lambda_1={lam1:.2f}$, $\lambda_2={lam2:.2f}$')

                # Plot derivative
                deriv = eval(f.__name__ + '_derivative')(fit.params, smooth_x_vec)
                ax11.plot(smooth_x_vec, deriv, color=ca_over_cb(colors1[i], colors2[j]), linestyle='--', linewidth=1,
                          alpha=0.4,
                          label=rf'$\lambda_1={lam1:.2f}$, $\lambda_2={lam2:.2f}$')

                # Save the fit parameters and score
                scores[f'{f.__name__},{lam1},{lam2}'] = (fit.params.valuesdict(), error(fit.params, f, smooth_x_vec))
                Z[i, j] = error(fit.params, f, smooth_x_vec)
            except:
                print(f'Fitting failed for lambda1 = {lam1}, lambda2 = {lam2}')

# find optimal model
optimal_model = min(scores.items(), key=lambda x: x[1][1])
opt_l1 = round(float(optimal_model[0].split(",")[1]), 2)
opt_l2 = round(float(optimal_model[0].split(",")[2]), 2)
print(
    f'Optimal model:\n\tlambda1 {optimal_model[0].split(",")[1]}, lambda2 {optimal_model[0].split(",")[2]}\n\tparameters {optimal_model[1][0]}\n\tscore {optimal_model[1][1]}')

# highlight optimal model
ax2.plot(smooth_x_vec, f(optimal_model[1][0], smooth_x_vec), color='blue', linestyle='--', linewidth=4,
         alpha=0.7,
         label='Optimal')
ax11.plot(smooth_x_vec, eval(f.__name__ + '_derivative')(optimal_model[1][0], smooth_x_vec), color='blue',
          linestyle='--', linewidth=4,
          alpha=0.7,
          label='Optimal')
optimal_pars = {k: round(v, 2) for k, v in optimal_model[1][0].items()}

# plot y=x
ax2.plot(smooth_x_vec, smooth_x_vec, color='black', linestyle='--', linewidth=1, label=r'$y=x$')
ax11.plot(smooth_x_vec, np.ones_like(smooth_x_vec), color='black', linestyle='--', linewidth=1)
# Plot data points
ax2.plot(x, y, 'r+', markersize=15, markeredgewidth=2, label='Data')

# colorbars
divider = make_axes_locatable(ax2)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=plt.cm.autumn_r,
                                orientation='vertical')
cb1.set_ticks([])
cb1.ax.text(0.5, 0.5, r'$\lambda_1$', ha='center', va='center', fontsize=20)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=plt.cm.summer_r,
                                orientation='vertical')
cb2.ax.text(0.5, 0.5, r'$\lambda_2$', ha='center', va='center', fontsize=20)

# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width, box.height * 0.9])
# Put a legend below current axis
leg = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=11)
for lh in leg.legendHandles:
    lh.set_alpha(1)

ax2.set_xlabel(r'Calcium $u$')
ax11.set_xlabel(r'Calcium $u$')

ax2.set_title('Fitted Gompertz', fontsize=20)
ax11.set_title('Gompertz derivatives', fontsize=20)

# plot error surface
surf = ax12.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, alpha=0.5)
# ax12.scatter(opt_l1, opt_l2, optimal_model[1][1], color='blue', s=150, marker='*')
ax12.text(opt_l1, opt_l2, optimal_model[1][1], fr'e={round(float(optimal_model[1][1]), 2)}', (1, 1, 0), fontsize=15,
          color='blue')
ax12.text(opt_l1, ax12.get_ylim()[0], ax12.get_zlim()[0], rf'$\lambda_1={opt_l1}$', 'x', fontsize=15, color='blue')
ax12.text(ax12.get_xlim()[0], opt_l2, ax12.get_zlim()[0], rf'$\lambda_2={opt_l2}$', 'y', fontsize=15, color='blue')
markerline, stemline, _ = ax12.stem([opt_l1], [opt_l2], [optimal_model[1][1]], orientation='y')
plt.setp(markerline, 'markerfacecolor', 'b')
plt.setp(stemline, 'color', 'b')
markerline, stemline, _ = ax12.stem([opt_l1], [opt_l2], [optimal_model[1][1]], orientation='x')
plt.setp(markerline, 'markerfacecolor', 'b')
plt.setp(stemline, 'color', 'b')
# invert x-axis and remove padding
ax12.set_xlim(1, 0)
ax12.set_ylim(0, 1)

ax12.set_xlabel(r'$\lambda_1$')
ax12.set_ylabel(r'$\lambda_2$')
ax12.set_title('Distance from desiderata', fontsize=20)

fig.suptitle(
    'Fits to Gompertz model\n' + f'Optimal model: $\lambda_1={opt_l1}, \lambda_2={opt_l2}$' + f'\nParameters {optimal_pars}\nError {round(optimal_model[1][1], 2)}',
    fontsize=30)
fig.show()

# display optimal model
fig2, (ax11, ax2) = plt.subplots(2, 1)
smooth_y_vec = f(optimal_model[1][0], smooth_x_vec)
ax11.plot(smooth_x_vec, smooth_y_vec, color='blue', linestyle='--', linewidth=1)
ax11.plot(smooth_x_vec, smooth_x_vec, color='black', linestyle='--', linewidth=1)

deriv = eval(f.__name__ + '_derivative')(optimal_model[1][0], smooth_x_vec)
ax2.plot(smooth_x_vec, deriv, color='blue', linestyle='--', linewidth=1)
ax2.plot(smooth_x_vec, np.ones_like(smooth_x_vec), color='black', linestyle='--', linewidth=1)

ax11.plot(x, y, 'r+', markersize=15, markeredgewidth=2, label='Data')
fig2.suptitle(
    f'Optimal model: $\lambda_1={opt_l1}, \lambda_2={opt_l2}$' + f'\nParameters {optimal_pars}\nError {round(optimal_model[1][1], 2)}')
fig2.tight_layout()
fig2.show()

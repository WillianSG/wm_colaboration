from lmfit import Minimizer, Parameters, report_fit
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap

x = np.array([0, 0.2, 0.3, 0.5, 0.7, 0.9, 1])
y = np.array([0, 0, 0.1, 0.5, 0.7, 0.9, 1])

fig, ((ax11, ax21)) = plt.subplots(2, 1, figsize=(25, 25))
ax11.plot(x, y, 'r+', markersize=15, markeredgewidth=2, label='Data')

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
    # calculate distance from f(x)=x as penalty term
    penalty1 = np.abs(model - x)
    # calculate distance from f'(x)=1 as penalty term
    penalty2 = np.abs(eval(func.__name__ + '_derivative')(params, x) - 1)

    return (penalty1, penalty2)


functions = [(gompertz, params_gompertz)]


# Write down the objective function that we want to minimize, i.e., the residuals
def residuals(params, func, x, data, lam1=0, lam2=0, exclude=[0, 1, 2, 3]):
    '''Model a logistic growth and subtract data'''
    # Logistic model
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
colors1 = plt.cm.copper_r(np.linspace(0, 2, lambdas))
colors2 = plt.cm.summer_r(np.linspace(0, 2, lambdas))

for f, params in functions:
    for i, lam1 in enumerate(X[0]):
        for j, lam2 in enumerate(Y[:, 0]):
            print(f'Fitting {f.__name__} with lambda1 = {lam1}, lambda2 = {lam2}')
            try:
                # Create a Minimizer object
                minner = Minimizer(residuals, params_gompertz, fcn_args=(f, x, y, lam1, lam2))
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
                ax11.plot(smooth_x_vec, smooth_y_vec, color=colors1[i] * colors2[j], linestyle='--', linewidth=1,
                          label=rf'$\lambda_1={lam1:.2f}$, $\lambda_2={lam2:.2f}$')

                # Plot derivative
                deriv = eval(f.__name__ + '_derivative')(fit.params, smooth_x_vec)
                ax21.plot(smooth_x_vec, deriv, color=colors1[i] * colors2[j], linestyle='--', linewidth=1)

                # Save the fit parameters and score
                scores[f'{f.__name__},{lam1},{lam2}'] = (fit.params.valuesdict(),
                                                         np.mean(penalty(smooth_y_vec, f, smooth_x_vec)))
                Z[i, j] = np.mean(penalty(smooth_y_vec, f, smooth_x_vec))
            except:
                print(f'Fitting failed for lambda1 = {lam1}, lambda2 = {lam2}')

# plot y=x
ax11.plot(smooth_x_vec, smooth_x_vec, color='black', linestyle='--', linewidth=1, label=r'$y=x$')
ax21.plot(smooth_x_vec, np.ones_like(smooth_x_vec), color='black', linestyle='--', linewidth=1)

# Shrink current axis's height by 10% on the bottom
box = ax21.get_position()
ax21.set_position([box.x0, box.y0 + box.height * 0.1,
                   box.width, box.height * 0.9])
# Put a legend below current axis
ax11.legend(loc='upper center', bbox_to_anchor=(0.5, -1.15), fancybox=True, shadow=True, ncol=10)

ax21.set_xlabel(r'Calcium $u$')

ax11.set_title('Fitted Gompertz')
ax21.set_title('Gompertz derivatives')

fig.suptitle(r'Gompertz and Richard logistic growth equation fitting for $f(u)$', fontsize=20)
# fig.tight_layout()
fig.show()

# plot error surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\lambda_1$')
ax.set_ylabel(r'$\lambda_2$')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# find optimal model
optimal_model = min(scores.items(), key=lambda x: x[1][1])
print(
    f'Optimal model:\n\tlambda1 {optimal_model[0].split(",")[1]}, lambda2 {optimal_model[0].split(",")[2]}\n\tparameters {optimal_model[1][0]}\n\tscore {optimal_model[1][1]}')
# display optimal model
fig, (ax11, ax21) = plt.subplots(2, 1)
smooth_y_vec = f(optimal_model[1][0], smooth_x_vec)
ax11.plot(smooth_x_vec, smooth_y_vec, color='orange', linestyle='--', linewidth=1)
ax11.plot(smooth_x_vec, smooth_x_vec, color='black', linestyle='--', linewidth=1)

deriv = eval(f.__name__ + '_derivative')(optimal_model[1][0], smooth_x_vec)
ax21.plot(smooth_x_vec, deriv, color='orange', linestyle='--', linewidth=1)
ax21.plot(smooth_x_vec, np.ones_like(smooth_x_vec), color='black', linestyle='--', linewidth=1)
fig.show()

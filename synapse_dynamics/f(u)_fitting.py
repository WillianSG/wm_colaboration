from lmfit import Minimizer, Parameters, report_fit
import numpy as np
import matplotlib.pylab as plt

x = np.array([0, 0.5, 0.7, 0.9, 1])
y = np.array([0, 0.1, 0.7, 0.9, 1])

plt.plot(x, y, 'r+', markersize=15, markeredgewidth=2, label='Data')

# Create object for parameter storing
params_gompertz = Parameters()
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params_gompertz.add_many(
    ('a', 1, False, None, None, None, None),
    ('b', 3, True, None, None, None, None),
    ('c', 7, True, None, None, None, None)
)  # I see it in the graph


# Write down the objective function that we want to minimize, i.e., the residuals


def residuals_gompertz(params, t, data):
    '''Model a logistic growth and subtract data'''
    # Get an ordered dictionary of parameter values
    v = params.valuesdict()
    # Logistic model
    model = v['a'] * np.exp(-np.exp(v['b'] - v['c'] * t))
    # Return residuals
    return model - data


# Create a Minimizer object
minner = Minimizer(residuals_gompertz, params_gompertz, fcn_args=(x, y))
# Perform the minimization
fit_gompertz = minner.minimize()

# Summarize results
report_fit(fit_gompertz)

result_gompertz = y + fit_gompertz.residual
plt.plot(x, result_gompertz, 'g.', markersize=15, label='Gompertz')
# Get a smooth curve by plugging a time vector to the fitted logistic model
t_vec = np.linspace(0, 1, 1000)
log_N_vec = np.ones(len(t_vec))
residual_smooth_gompertz = residuals_gompertz(fit_gompertz.params, t_vec, log_N_vec)
plt.plot(t_vec, residual_smooth_gompertz + log_N_vec, 'g', linestyle='--', linewidth=1)

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


# Define the model
def residuals_genlogistic(params, t, data):
    '''Model a logistic growth and subtract data'''
    # Get an ordered dictionary of parameter values
    v = params.valuesdict()
    # Logistic model
    model = v['A'] + (v['K'] - v['A']) / (1 + v['Q'] * np.exp(-v['B'] * (t - v['T']))) ** (1 / v['mu'])
    # Return residuals
    return model - data


# Perform the fit
# Create a Minimizer object
minner = Minimizer(residuals_genlogistic, params_genlogistic, fcn_args=(x, y))

# Perform the minimization
fit_genlogistic = minner.minimize()

# Summarize results
report_fit(fit_genlogistic)

result_genlogistic = y + fit_genlogistic.residual
plt.plot(x, result_genlogistic, 'b.', markersize=15, label='Richard')
# Get a smooth curve by plugging a time vector to the fitted logistic model
t_vec = np.linspace(0, 1, 1000)
log_N_vec = np.ones(len(t_vec))
residual_smooth_genlogistic = residuals_genlogistic(fit_genlogistic.params, t_vec, log_N_vec)
plt.plot(t_vec, residual_smooth_genlogistic + log_N_vec, 'b', linestyle='--', linewidth=1)

plt.legend()
plt.show()

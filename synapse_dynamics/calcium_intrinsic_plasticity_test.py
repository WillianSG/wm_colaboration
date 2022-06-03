from brian2 import NeuronGroup, Synapses, StateMonitor, run, ms, second, mV, SpikeMonitor, PoissonGroup, Hz, Network, \
    collect, store, restore, volt
import matplotlib.pyplot as plt
import numpy as np
from helper_functions.other import *
import sympy

w = 10 * mV
Vth_e_init = -52 * mV
Vr_e = -65 * mV
taum_e = 20 * ms
tau_epsp = 3.5 * ms
Vrst_e = -65 * mV

U = 0.2
tau_f = 0.7 * second
tau_d = 90 * ms
Vth_e_decr = 0.14 * mV
tau_Vth_e = 0.1 * second
k = 10
mu = 0.7

th_linear = 'dVth_e/dt = ((Vth_e_init - 0.002 * u * volt) - Vth_e) / tau_Vth_e : volt'
th_logistic = f'dVth_e/dt = (Vth_e_init - (0.002 / (1 + exp(-{k}*(u-{mu}))) * volt) - Vth_e) / tau_Vth_e : volt'
th_sigmoid_logit = f'dVth_e/dt = (Vth_e_init - (0.002 * (1 + ((u * (1 - 0.5)) / (0.5 * (1 - u)))**-{k} )**-1 * volt) - Vth_e) / tau_Vth_e  : volt'

# -- choose which calcium-threshold function to use
f_u = th_logistic

f_string = f_u.replace(' : ', ' = ').split(' = ')[1].replace('- Vth_e', '').replace('exp', 'np.exp')
f_string_latex = sympy.latex(sympy.sympify(f_string.replace('* volt', '').replace('np.exp', 'exp')))

E_model_old = '''
                dVepsp/dt = -Vepsp / tau_epsp : volt
                dVm/dt = (Vepsp - (Vm - Vr_e)) / taum_e : volt
                du/dt = ((U - u) / tau_f) : 1
                dVth_e/dt = (Vth_e_init - Vth_e) / tau_Vth_e : volt
                '''
E_reset_old = '''
                Vm = Vrst_e 
                Vth_e -= Vth_e_decr
                u = u + U * (1 - u)
                '''
E_model_new = '''
                dVepsp/dt = -Vepsp / tau_epsp : volt
                dVm/dt = (Vepsp - (Vm - Vr_e)) / taum_e : volt
                du/dt = ((U - u) / tau_f) : 1
                ''' + f_u
E_reset_new = '''
                Vm = Vrst_e 
                u = u + U * (1 - u)
                '''

E = NeuronGroup(1,
                model=E_model_new,
                reset=E_reset_new,
                threshold='Vm > Vth_e',
                method='euler')
E.Vm = Vrst_e
E.Vth_e = Vth_e_init
E.u = U

S = Synapses(E, E,
             model='''
            dx_/dt = ((1 - x_) / tau_d) : 1 (clock-driven)
             ''',
             on_pre='''
              x_ = x_ - u * x_
              Vepsp += w * ((x_ * u_pre) / U)
            ''',
             method='euler')
S.connect()

spikemon = SpikeMonitor(E)
E_mon = StateMonitor(E, ['u', 'Vm', 'Vth_e'], record=True)
S_mon = StateMonitor(S, ['x_'], record=True)

store()


def run_firing_at_hz(start, stop, hz):
    start = start * second
    stop = stop * second

    store()

    t_firing = np.arange(start, stop, hz2sec(hz))
    for _ in t_firing:
        restore()
        E.Vm = 0 * mV
        if len(t_firing) <= 1:
            run(stop - start)
        else:
            run(hz2sec(hz))
        store()


def cyclic_pss(start, stop, jitter=0.0):
    t_jitter = np.random.normal(loc=0, scale=jitter, size=3)

    t = start + t_jitter[0]
    while t < stop:
        run_firing_at_hz(t, t + 0.55 + t_jitter[1], 1)
        run_firing_at_hz(t + 0.55 + t_jitter[1], t + 0.6 + t_jitter[1], 120)
        t = t + 0.6 + t_jitter[1] + t_jitter[2]


# run 1 second of silence, then simulate GS, then PSS for 4 seconds, then 4 of silence
run(1 * second)
run_firing_at_hz(4, 4.1, 155)
cyclic_pss(4.1, 8.1)
run(4 * second)

# plot results of simulation
fig, ax1 = plt.subplots(figsize=(10, 5))
ax_twin = ax1.twinx()
ax1.plot(E_mon.t / second, E_mon.Vth_e[0, :], color='g', label='Vth_e')
ax1.set_ylabel('Vth_e (V)', c='g')
ax1.tick_params(axis='y', colors='g')
ax_twin.plot(E_mon.t / second, E_mon.u[0, :], color='b', label='u')
ax_twin.set_ylabel('u', c='b')
ax_twin.tick_params(axis='y', colors='b')
ax_twin.set_ylim(0, 1)
# ax_twin.plot(S_mon.t / second, S_mon.x_[0, :], color='r', label='x')
for t in spikemon.t:
    plt.axvline(t / second, c='C1', alpha=0.2)
ax1.legend([ax1.lines[0], ax_twin.lines[0], ax_twin.lines[1]], ['Vth_e', 'u', 'spikes'])
# ax1.set_ylim([-60 * mV, -50 * mV])
ax1.set_xlabel('Time (s)')
fig.suptitle('Simulated firing pattern of attractor with STSP')
plt.show()

# plot f(u) function used
u = np.arange(0, 1, 0.01)
f = eval(f_string) * 100
plt.plot(u, f, c='y')
plt.ylabel('f(u) (mV)')
plt.title('f(u) function')
plt.text(1, np.mean(f), fr'${f_string_latex}$', color="y", fontsize=24, horizontalalignment="right",
         verticalalignment="top")
plt.tight_layout()
plt.show()

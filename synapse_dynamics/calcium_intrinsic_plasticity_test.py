from brian2 import NeuronGroup, Synapses, StateMonitor, run, ms, second, mV, SpikeMonitor, PoissonGroup, Hz, Network, \
    collect, store, restore
import matplotlib.pyplot as plt
import numpy as np
from helper_functions.other import *

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
tau_Vth_e = 1.8 * second

E_model_old = '''
                dVepsp/dt = -Vepsp / tau_epsp : volt
                dVm/dt = (Vepsp - (Vm - Vr_e)) / taum_e : volt
                dVth_e/dt = (Vth_e_init - Vth_e) / tau_Vth_e : volt
                du/dt = ((U - u) / tau_f) : 1
                '''
E_reset_old = '''
                Vm = Vrst_e 
                Vth_e -= Vth_e_decr
                u = u + U * (1 - u)
                '''
E_model_new = '''
                dVepsp/dt = -Vepsp / tau_epsp : volt
                dVm/dt = (Vepsp - (Vm - Vr_e)) / taum_e : volt
                dVth_e/dt = ((Vth_e_init - 0.002 * u * volt) - Vth_e) / tau_Vth_e : volt
                du/dt = ((U - u) / tau_f) : 1
                '''
E_reset_new = '''
                Vm = Vrst_e 
                u = u + U * (1 - u)
                '''

E = NeuronGroup(1,
                model=E_model_new,
                reset=E_reset_new,
                threshold='Vm > Vth_e',
                method='linear')
E.Vth_e = Vth_e_init

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

    t_firing = np.arange(start, stop, hz2sec(hz))
    for _ in t_firing:
        restore()
        E.Vm = 0 * mV
        if len(t_firing) <= 1:
            run(stop - start)
        else:
            run(hz2sec(hz))
        store()


def cyclic_pss(start, stop):
    t = start
    while t < stop:
        run_firing_at_hz(t, t + 0.55, 1)
        run_firing_at_hz(t + 0.55, t + 0.6, 120)
        t += 0.6


run_firing_at_hz(0, 0.1, 155)
cyclic_pss(0.1, 4)
run(4 * second)
# run_firing_at_hz(0.7, 1, 1)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax_twin = ax1.twinx()
# plt.plot(spikemon.t / ms, spikemon.i, '.k')
# plt.show()
ax1.plot(E_mon.t / second, E_mon.Vth_e[0, :], color='g', label='Vth_e')
ax1.set_ylabel('Vth_e (V)')
ax_twin.plot(E_mon.t / second, E_mon.u[0, :], color='b', label='u')
ax_twin.set_ylabel('u, x')
ax_twin.plot(S_mon.t / second, S_mon.x_[0, :], color='r', label='x')
for t in spikemon.t:
    plt.axvline(t / second, ls='--', c='C1', alpha=0.6)
ax1.legend([ax1.lines[0], ax_twin.lines[0], ax_twin.lines[1]], ['Vth_e', 'u', 'x'])
# ax1.set_ylim([-60 * mV, -50 * mV])
ax1.set_xlabel('Time (s)')
fig.suptitle('Simulated firing pattern of attractor with STSP')
plt.show()

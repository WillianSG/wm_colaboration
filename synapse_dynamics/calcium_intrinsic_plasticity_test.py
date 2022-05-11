from brian2 import NeuronGroup, Synapses, StateMonitor, run, ms, second, mV, SpikeMonitor, PoissonGroup, Hz
import matplotlib.pyplot as plt
import numpy as np

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

E = NeuronGroup(1,
                model='''
                        dVepsp/dt = -Vepsp / tau_epsp : volt
                        dVm/dt = (Vepsp - (Vm - Vr_e)) / taum_e : volt
                        dVth_e/dt = (Vth_e_init - Vth_e) / tau_Vth_e : volt
                        du/dt = ((U - u) / tau_f) : 1
                        ''',
                reset='''
                Vm = Vrst_e 
                Vth_e -= Vth_e_decr
                u = u + U * (1 - u)
                ''',
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

E.run_regularly('Vm = 0 *mV', dt=400 * ms)
run(4 * second)

fig, ax1 = plt.subplots()
ax_twin = ax1.twinx()
# plt.plot(spikemon.t / ms, spikemon.i, '.k')
# plt.show()
ax1.plot(E_mon.t / second, E_mon.Vth_e[0, :], color='g', label='Vth_e')
ax1.set_ylabel('Vth_e (V)')
ax_twin.plot(E_mon.t / second, E_mon.u[0, :], color='b', label='u')
ax_twin.set_ylabel('u, x')
ax_twin.plot(S_mon.t / second, S_mon.x_[0, :], color='r', label='x')
for t in spikemon.t:
    plt.axvline(t / second, ls='--', c='C1', lw=3)
ax1.legend([ax1.lines[0], ax_twin.lines[0], ax_twin.lines[1]], ['Vth_e', 'u', 'x'])
ax1.set_xlabel('Time (s)')
plt.show()

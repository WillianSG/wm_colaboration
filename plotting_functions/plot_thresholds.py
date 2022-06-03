from brian2 import volt, second, mV
import numpy as np
import matplotlib.pyplot as plt


def plot_thresholds(rcn, attractors, show=True):
    fig, (ax11, ax21) = plt.subplots(2, figsize=(10, 5))

    Vth_e_init = rcn.Vth_e_init
    u_a1 = np.mean(rcn.E_rec.u[attractors[0][1], :], axis=0)
    Vth_a1 = np.mean(rcn.E_rec.Vth_e[attractors[0][1], :], axis=0)
    f_u_a1 = (Vth_e_init - 0.002 * u_a1 * volt)
    l11 = ax11.plot(rcn.E_rec.t / second, f_u_a1, c='y', label='f(u)')
    ax11.tick_params(axis='y', labelcolor='y')
    ax12 = ax11.twinx()
    l12 = ax12.plot(rcn.E_rec.t / second, Vth_a1, c='g', label='u')
    ax12.tick_params(axis='y', labelcolor='g')
    ax11.set_title('Attractor 1')

    u_a2 = np.mean(rcn.E_rec.u[attractors[1][1], :], axis=0)
    Vth_a2 = np.mean(rcn.E_rec.Vth_e[attractors[1][1], :], axis=0)
    f_u_a2 = (Vth_e_init - 0.002 * u_a2 * volt)
    l21 = ax21.plot(rcn.E_rec.t / second, f_u_a2, c='y', label='f(u)')
    ax21.tick_params(axis='y', labelcolor='y')
    ax22 = ax21.twinx()
    l22 = ax22.plot(rcn.E_rec.t / second, Vth_a2, c='g', label='u')
    ax22.tick_params(axis='y', labelcolor='g')
    ax21.set_title('Attractor 2')

    stacked_arrays = np.hstack((Vth_a1, Vth_a2, f_u_a1, f_u_a2))
    ax11.set_ylim([np.min(stacked_arrays) * volt, -52 * mV])
    ax12.set_ylim(ax11.get_ylim())
    ax21.set_ylim(ax11.get_ylim())
    ax22.set_ylim(ax11.get_ylim())

    fig.legend((l11[0], l12[0]), ('f(u)', 'Vth'), loc='upper right')
    fig.suptitle('Asymptotic value of thresholds')

    fig.tight_layout()

    if show:
        fig.show()

    return fig

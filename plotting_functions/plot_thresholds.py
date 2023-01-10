from brian2 import volt, second, mV
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_thresholds(path, file_name, rcn, attractors, show=True):
    fig, axes = plt.subplots(3, 2, figsize=(20, 10), gridspec_kw={'height_ratios': [1, 1, 0.2]})

    Vth_e_init = rcn.Vth_e_init
    for atr, ax in zip(attractors, axes):
        u = rcn.E_rec.u[atr[1], :]
        f_u = (Vth_e_init - 0.002 * u * volt)
        f_u_mean = np.mean(f_u, axis=0)
        f_u_std = np.std(f_u, axis=0)
        f_u_min = np.min(f_u, axis=0)
        f_u_max = np.max(f_u, axis=0)
        l11 = ax[0].plot(rcn.E_rec.t / second, f_u_mean, c='y', label='f(u)')
        
        ax[0].fill_between(
            rcn.E_rec.t / second,
            f_u_mean + f_u_std,
            f_u_mean - f_u_std,
            color='y',
            alpha=0.4)
        ax[0].fill_between(
            rcn.E_rec.t / second,
            f_u_max,
            f_u_min,
            color='y',
            alpha=0.2)
        
        ax[0].tick_params(axis='y', labelcolor='y')
        
        ax0twin = ax[0].twinx()
        
        Vth = rcn.E_rec.Vth_e[atr[1], :]
        Vth_mean = np.mean(Vth, axis=0)
        Vth_std = np.std(Vth, axis=0)
        Vth_min = np.min(Vth, axis=0)
        Vth_max = np.max(Vth, axis=0)
        l11twin = ax0twin.plot(rcn.E_rec.t / second, Vth_mean, c='g', label='u')
        
        ax0twin.fill_between(
            rcn.E_rec.t / second,
            Vth_mean + Vth_std,
            Vth_mean - Vth_std,
            color='g',
            alpha=0.4)
        ax0twin.fill_between(
            rcn.E_rec.t / second,
            Vth_min,
            Vth_max,
            color='g',
            alpha=0.2)
        
        ax0twin.tick_params(axis='y', labelcolor='g')
        
        ax[0].set_title(f'Attractor {atr[0]}')

        Vm = rcn.E_rec.Vm[atr[1], :]
        Vm_mean = np.mean(Vm, axis=0)
        Vm_std = np.std(Vm, axis=0)
        Vm_min = np.min(Vm, axis=0)
        Vm_max = np.max(Vm, axis=0)
        
        l12 = ax[1].plot(rcn.E_rec.t / second, Vm_mean, c='r', label='Vm')
        
        ax[1].fill_between(
            rcn.E_rec.t / second,
            Vm_mean + Vm_std,
            Vm_mean - Vm_std,
            color='r',
            alpha=0.4)
        ax[1].fill_between(
            rcn.E_rec.t / second,
            Vm_min,
            Vm_max,
            color='r',
            alpha=0.2)
        
        ax[1].tick_params(axis='y', labelcolor='r')
        
        ax1twin = ax[1].twinx()
        
        l12twin = ax1twin.plot(rcn.E_rec.t / second, Vth_mean, c='g', label='u')
        
        ax1twin.fill_between(
            rcn.E_rec.t / second,
            Vth_mean + Vth_std,
            Vth_mean - Vth_std,
            color='g',
            alpha=0.4)
        ax1twin.fill_between(
            rcn.E_rec.t / second,
            Vth_min,
            Vth_max,
            color='g',
            alpha=0.2)
        
        ax1twin.tick_params(axis='y', labelcolor='g')
        ax[1].set_title(f'Attractor {atr[0]}')

        stacked_arrays_1 = np.hstack((Vth, f_u))
        ax[0].set_ylim([np.min(stacked_arrays_1) * volt, -52 * mV])
        ax0twin.set_ylim(ax[0].get_ylim())

        stacked_arrays_2 = np.hstack((Vth, Vm))
        ax[1].set_ylim([np.min(stacked_arrays_2) * volt, -52 * mV])
        ax1twin.set_ylim(ax[1].get_ylim())

    fig.legend((l11[0], l11twin[0]), ('f(u)', 'Vth'), loc='upper left')
    fig.legend((l12twin[0], l12[0]), ('Vth', 'Vm'), loc='upper right')
    fig.suptitle('Asymptotic value of thresholds')

    axes[2, 0].set_axis_off()
    axes[2, 1].set_axis_off()
    plt.figtext(0.5, 0.02,
                'Filled line: mean value\nDark shading: 1 standard deviation\nLight shading: exteme values\n\nWhen Vm > Vth then a spike is generated',
                fontsize=14, ha='center')

    fig.tight_layout()

    if show:
        fig.show()

    fig.savefig(os.path.join(path, file_name + '.png'))

    return fig

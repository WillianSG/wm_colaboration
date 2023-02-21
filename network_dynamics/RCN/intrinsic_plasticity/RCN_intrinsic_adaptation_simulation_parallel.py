import glob
import atexit
import pickle
import re
import shutil
import signal
import pandas as pd

from matplotlib.offsetbox import AnchoredText

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from brian2 import prefs, ms, Hz, mV, second
from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from helper_functions.other import *
from plotting_functions.x_u_spks_from_basin import plot_x_u_spks_from_basin
from plotting_functions.rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
from plotting_functions.spike_synchronisation import *
from plotting_functions.plot_thresholds import *
from helper_functions.population_spikes import count_ps
from helper_functions.population_spikes import count_ps


def cleanup(exit_code=None, frame=None):
    print("Cleaning up leftover files...")
    # delete tmp folder
    shutil.rmtree(tmp_folder)


atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# make tmp folder with timestamp
tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(tmp_folder)


class RCN_Estimator(BaseEstimator, RegressorMixin):
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __init__(self, ba=15, i_e_w=10, i_freq=20, cue_percentage=100, a2_cue_time=0.1, attractors=2):
        self.rcn = RecurrentCompetitiveNet(
            plasticity_rule='LR4',
            parameter_set='2.2')

        self.ba = ba
        self.i_e_w = i_e_w
        self.i_freq = i_freq
        self.cue_percentage = cue_percentage
        self.a2_cue_time = a2_cue_time
        self.attractors = attractors

        plastic_syn = False
        plastic_ux = True
        self.rcn.E_E_syn_matrix_snapshot = False
        self.rcn.w_e_i = 3 * mV  # for param. 2.1: 5*mV
        self.rcn.w_max = 10 * mV  # for param. 2.1: 10*mV
        self.rcn.spont_rate = self.ba * Hz

        self.rcn.w_e_i = 3 * mV  # 3 mV default
        self.rcn.w_i_e = self.i_e_w * mV  # 1 mV default

        # -- intrinsic plasticity setup (Vth_e_decr for naive / tau_Vth_e for calcium-based)

        self.rcn.tau_Vth_e = 0.1 * second  # 0.1 s default
        # self.rcn.Vth_e_init = -51.6 * mV           # -52 mV default
        self.rcn.k = 4  # 2 default

        self.rcn.net_init()

        # -- synaptic augmentation setup
        self.rcn.U = 0.2  # 0.2 default

        self.rcn.set_stimulus_i(
            stimulus='flat_to_I',
            frequency=self.i_freq * Hz)  # default: 15 Hz

        # --folder for simulation results
        # save_dir = os.path.join(os.path.join(timestamp_folder, f'BA_{ba}_GS_{gs_percentage}_W_{i_e_w}_I_{i_freq}'))
        # os.mkdir(save_dir)
        # self.rcn.net_sim_data_path = save_dir

        # output .txt with simulation summary.
        # _f = os.path.join(self.rcn.net_sim_data_path, 'simulation_summary.txt')

        # self.attractors = []

        if self.attractors >= 1:
            self.stim1_ids = self.rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=0)
            self.rcn.set_potentiated_synapses(self.stim1_ids)
            # A1 = list(range(0, 64))
            # self.attractors.append(('A1', A1))
        if self.attractors >= 2:
            self.stim2_ids = self.rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=100)
            self.rcn.set_potentiated_synapses(self.stim2_ids)
            # A2 = list(range(100, 164))
            # self.attractors.append(('A2', A2))
        if self.attractors >= 3:
            self.stim3_ids = self.rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=180)
            self.rcn.set_potentiated_synapses(self.stim3_ids)
            # A3 = list(range(180, 244))
            # self.attractors.append(('A3', A3))

        self.rcn.set_E_E_plastic(plastic=plastic_syn)
        self.rcn.set_E_E_ux_vars_plastic(plastic=plastic_ux)

    def fit(self, X, y=None):
        # stimulation_amount = []
        self.attractors_t_windows = []

        # cue attractor 1
        gs_A1 = (self.cue_percentage, self.stim1_ids, (0, 0.1))
        act_ids = self.rcn.generic_stimulus(
            frequency=self.rcn.stim_freq_e,
            stim_perc=gs_A1[0],
            subset=gs_A1[1])
        # stimulation_amount.append(
        #     (100 * np.intersect1d(act_ids, stim1_ids).size / len(stim1_ids),
        #      100 * np.intersect1d(act_ids, stim2_ids).size / len(stim2_ids))
        # )
        self.rcn.run_net(duration=gs_A1[2][1] - gs_A1[2][0])
        self.rcn.generic_stimulus_off(act_ids)

        # wait for 2 seconds before cueing second attractor
        self.rcn.run_net(duration=2)

        # cue attractor 2
        gs_A2 = (self.cue_percentage, self.stim2_ids, (2.1, 2.1 + self.a2_cue_time))
        act_ids = self.rcn.generic_stimulus(
            frequency=self.rcn.stim_freq_e,
            stim_perc=gs_A2[0],
            subset=gs_A2[1])
        # stimulation_amount.append(
        #     (100 * np.intersect1d(act_ids, stim1_ids).size / len(stim1_ids),
        #      100 * np.intersect1d(act_ids, stim2_ids).size / len(stim2_ids))
        # )
        self.rcn.run_net(duration=gs_A2[2][1] - gs_A2[2][0])
        self.rcn.generic_stimulus_off(act_ids)

        # wait for another 2 seconds before ending
        self.rcn.run_net(duration=2 + self.a2_cue_time)

        # log period of independent activity for attractor 1.
        self.attractors_t_windows.append((gs_A1[2][1], gs_A2[2][0], gs_A1[2]))

        # log period of independent activity for attractor 2.
        self.attractors_t_windows.append((gs_A2[2][1], self.rcn.E_E_rec.t[-1] / second, gs_A2[2]))

        self.results_ready = True

        return self

    def predict(self, X):
        assert self.results_ready, "You must call fit() before calling predict()"

        return np.array([0])

    def score(self, X, y=None, sample_weight=None):
        assert self.results_ready, "You must call fit() before calling score()"

        atr_ps_counts = count_ps(rcn=self.rcn, attractor_cueing_order=)

        # -- count reactivations
        trig = 0
        spont = 0
        for k, v in atr_ps_counts.items():
            for k, v in v.items():
                if k == 'triggered':
                    trig += v
                if k == 'spontaneous':
                    spont += v

        # TODO do we like this way of scoring?
        return trig / (trig + spont)

    def plot_1(self):
        assert self.results_ready, "You must call fit() before calling plot()"

        title_addition = f'BA {ba} Hz, GS {gs_percentage} %, I-to-E {i_e_w} mV, I input {i_freq} Hz'
        filename_addition = f'_BA_{ba}_GS_{gs_percentage}_W_{i_e_w}_Hz_{i_freq}'

        gss = [gs_A1, gs_A2]
        fig = plot_x_u_spks_from_basin(
            path=save_dir,
            filename='x_u_spks_from_basin' + filename_addition,
            title_addition=title_addition,
            generic_stimuli=gss,
            rcn=rcn,
            attractors=attractors,
            num_neurons=len(rcn.E),
            show=args.show)
        # plot_syn_matrix_heatmap( path_to_data=rcn.E_E_syn_matrix_path )

        return fig

    def plot_2(self):
        assert self.results_ready, "You must call fit() before calling plot()"

        fig = plot_rcn_spiketrains_histograms(
            Einp_spks=rcn.get_Einp_spks()[0],
            Einp_ids=rcn.get_Einp_spks()[1],
            stim_E_size=rcn.stim_size_e,
            E_pop_size=rcn.N_input_e,
            Iinp_spks=rcn.get_Iinp_spks()[0],
            Iinp_ids=rcn.get_Iinp_spks()[1],
            stim_I_size=rcn.stim_size_i,
            I_pop_size=rcn.N_input_i,
            E_spks=rcn.get_E_spks()[0],
            E_ids=rcn.get_E_spks()[1],
            I_spks=rcn.get_I_spks()[0],
            I_ids=rcn.get_I_spks()[1],
            t_run=rcn.net.t,
            path=save_dir,
            filename='rcn_population_spiking' + filename_addition,
            title_addition=title_addition,
            show=args.show)

        return fig


num_par = 2
cv = 2
experiment = 'cueing'
# -- define grid search parameters
if experiment == 'params':
    param_grid = [{}, {}]
    fine_search = False
elif experiment == 'cueing':
    param_grid = {
        'a2_cue_time': np.linspace(0.1, 1, num=np.rint(num_par).astype(int)),
    }
    fine_search = True

estimate_search_time(RCN_Estimator(), param_grid, cv, 2 if fine_search else 1)

print('Initial search')
print('Param grid:', param_grid)
gs = GridSearchCV(RCN_Estimator(), param_grid=param_grid, n_jobs=1, verbose=3, cv=cv)
gs.fit(np.zeros(cv))
print('Best parameters:', gs.best_params_)
print('Best score:', gs.best_score_)
pd_results = pd.DataFrame(gs.cv_results_)
pd_results.to_csv('./RCN_grid_search_results.csv')

estimator = RCN_Estimator(**gs.best_params_)
estimator.fit([0])
print('Score on test run', estimator.score([0]))
estimator.plot().show()

if fine_search:
    # -- run grid search again in a neighbourhood of the best parameters
    param_grid_fine = {k: np.random.normal(v, v * 0.15, size=num_par)
                       for k, v in gs.best_params_.items()}

    print('Fine search')
    gs_fine = GridSearchCV(RCN_Estimator(), param_grid=param_grid_fine, n_jobs=-1, verbose=3, cv=cv)
    gs_fine.fit(np.zeros(cv))
    print('Best parameters:', gs_fine.best_params_)
    print('Best score:', gs_fine.best_score_)
    pd_results_fine = pd.DataFrame(gs_fine.cv_results_)
    pd_results_fine.to_csv('./RCN_grid_search_fine_results.csv')

    estimator = RCN_Estimator(**gs_fine.best_params_)
    estimator.fit([0])
    print('Score on test run', estimator.score([0]))
    estimator.plot().show()

size_L = 10
size_M = 8
size_S = 6
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
fig.set_size_inches((3.5, 3.5 * ((5. ** 0.5 - 1.0) / 2.0)))
# fig.suptitle(exp_name, fontsize=size_L)
ax.set_ylabel('Triggered reactivation ratio', fontsize=size_M)
ax.set_xlabel('Cue length (s)', fontsize=size_M)
ax.tick_params(axis='x', labelsize=size_S)
ax.tick_params(axis='y', labelsize=size_S)

ax.plot(pd_results['param_plot'], pd_results['mean_test_score'], c='g')
ax.plot(pd_results['param_plot'], pd_results['mean_test_score'] - pd_results['std_test_score'],
        linestyle="--", alpha=0.5, c="g")
ax.plot(pd_results['param_plot'], pd_results['mean_test_score'] + pd_results['std_test_score'],
        linestyle="--", alpha=0.5, c="g")
ax.fill_between(pd_results['param_plot'],
                pd_results['mean_test_score'] - pd_results['std_test_score'],
                pd_results['mean_test_score'] + pd_results['std_test_score'],
                alpha=0.3, color="g")
fig.tight_layout()
fig.show()

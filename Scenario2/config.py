# Model co-existence Config

import itertools


class Config:
    # Dataset declarations
    simulations = {
        'MIMICDeathAKI': {
            'dataset': '../Datasets/MIMIC/MIMIC_Death_AKI.pickle',
            'display_name': 'MIMIC IV: AKI prediction → Mortality prediction',
            'first_oc': 'aki',
            'second_oc': 'death',
            'admit_col': 'icu_admin_time',
            'td_col': 'aki_to_death',
        },
        'SINAIDeathAKI': {
            'dataset': '../Datasets/SINAI/SINAI_DeathAKI.pickle',
            'display_name': 'MSHS: AKI prediction → Mortality prediction',
            'first_oc': 'AKI',
            'second_oc': 'DEATH',
            'admit_col': 'ICU_STAY_START',
            'td_col': 'DEATH_AKI_TD',
        }
    }

    # General
    oc_diff_time_limit = 3  # Days between OC1 and OC2 at most

    # Common to all simulations
    effect_sizes = [0, .05, .1, .2, .5, .75, 1]  # This fraction of patients has a manifest effect
    plot_effect_sizes = [0, .05, .1, .2, .5, .75]
    random_state = 42

    dir_results = 'Results/'
    dir_plots = 'Plots/'

    # Model
    run_params_default = {
        'model_name': 'XGB',
        'first_oc_train_interval': .50,  # Decimal indicates perc of total data collection
        'downsample_dataset': False,  # This occurs with respect to outcome 2
        'hyperparameter_optimization': True,
        'conversions_due_to_neglect': True,
        'threshold_calibration': '90SENS',  # Possible options: 'YOUDEN', '90SENS', '90SPEC'
        'mitigation_strategy': None,  # None, 'FEATURE', 'DROP'
    }

    neglect_cap = {0: 0, 'default': .01}  # Effect size specific to neglected patients
    additional_positives_scale_factor = 1.2  # Minimum: 1 | This many additional positive predictions need
                                        # to be tracked as people who have had a prediction implemented


def get_run_params():
    # CRITICAL: These names MUST match the names in the config file
    run_params_pos = {
        'model_name': ['XGB', 'LASSO'],
        'first_oc_train_interval': [.25, .50],
        'downsample_dataset': [True, False],
        'hyperparameter_optimization': [False, True],
        'conversions_due_to_neglect': [True, False],
        'threshold_calibration': ['90SENS', '90SPEC'],
        'mitigation_strategy': [None, 'FEATURE', 'DROP'],
    }

    all_run_params = []
    for run_params in (run_params_pos,):
        for values in itertools.product(*map(run_params.get, run_params.keys())):
            all_run_params.append(dict(zip(run_params.keys(), values)))

    return all_run_params

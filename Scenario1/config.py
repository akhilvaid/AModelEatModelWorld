# Config for the Retrainer

import itertools


class Config:
    # NOTE: 'INTERVAL' gets dropped in each sim
    simulations = {
        'MIMICDeath': {
            'dataset': '../Datasets/MIMIC/MIMIC_Death_AKI.pickle',
            'display_name': 'MIMIC IV: Mortality prediction',  # For plotting
            'divisions': 10,  # This many steps / divisions of the time variable
            'time_var': 'icu_admin_time',  # Split along this variable
            'outcome': 'death',
            'model_prevents_outcome': True,  # Is the model trying to prevent predicted outcome or encourage it
            'not_for_training': ['INTERVAL', 'aki', 'aki_to_death', 'icu_admin_time']  # Drop these columns from training
        },
        'SINAIPostCathAKI': {
            'dataset': '../Datasets/SINAI/SINAI_PostCathAKI.pickle',
            'display_name': 'MSHS: AKI following cardiac catheterization',  # For plotting
            'divisions': 10,  # This many steps / divisions of the time variable
            'time_var': 'CATH_PROC_START_TIME',  # Split along this variable
            'outcome': 'AKI',
            'model_prevents_outcome': True,  # Is the model trying to prevent predicted outcome or encourage it
            'not_for_training': ['INTERVAL', 'CATH_PROC_START_TIME']
        }
    }

    # Specific to simulation if called without any params from __main__
    run_params_default = {
        'model_name': 'XGB',
        'downsample_dataset': False,
        'conversions_due_to_neglect': True,
        'threshold_calibration': '90SENS',  # Possible options: '90SENS', '90SPEC' and possibly 'YOUDEN'
        'shuffle_admissions': False,  # Shuffle admission times to ensure no bias due to new treatments
        'mitigation_strategy': None,  # None, 'FEATURE', 'DROP'
        'hyperparameter_optimization': True,
    }

    # Common to all simulation RUNS
    effect_sizes = [0, .05, .1, .2, .5, .75, 1]  # This fraction of patients has a manifest effect
    plot_effect_sizes = [0, .05, .1, .2, .5, .75]
    random_state = 42

    dir_results = 'Results'
    dir_plots = 'Plots'

    neglect_cap = {0: 0, 'default': .01}  # Effect size specific to neglected patients
    additional_positives_scale_factor = 1.2     # Minimum: 1 | This many additional positive predictions need
                                                # to be tracked as people who have had a prediction implemented

    # General debug switch
    debug = False

    # Save errors here
    error_file = 'Errors'


# Generate a list of all possible combinations of parameters
def get_run_params():
    # CRITICAL: These names MUST match the names in the config file
    run_params_pos = {
        'model_name': ['XGB', 'LASSO'],
        'downsample_dataset': [True, False],
        'conversions_due_to_neglect': [True, False],
        'threshold_calibration': ['90SENS', '90SPEC'],
        'shuffle_admissions': [True, False],
        'mitigation_strategy': [None, 'FEATURE', 'DROP'],
        'hyperparameter_optimization': [True, False],
    }

    all_run_params = []
    for run_params in (run_params_pos,):
        for values in itertools.product(*map(run_params.get, run_params.keys())):
            all_run_params.append(dict(zip(run_params.keys(), values)))

    return all_run_params

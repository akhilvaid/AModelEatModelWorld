# Model retraining
# For future reference
# sim_params is for the "simulations" dict in the Config
# run_params is for each individual run

import os
import sys
import copy
import warnings

import tqdm
import pandas as pd
import numpy as np

from scipy.spatial import distance

from config import Config
sys.path.append('../Utils/'); from helper_functions import HelperFunctions, is_sim_valid  # Just disgusting


class Trainer:
    def __init__(self, simulation_name, run_params=None, parallel=False):
        self.simulation_name = simulation_name
        assert '_' not in simulation_name, 'Simulation name cannot contain underscores'

        sim_params = Config.simulations[simulation_name]  # Load training params from the config
        self.dataset = pd.read_pickle(sim_params['dataset'])  # Patient identifiers HAVE to be in the index
        self.dataset_div = sim_params['divisions']
        self.time_var = sim_params['time_var']
        self.outcome = sim_params['outcome']
        self.model_prevents_outcome = sim_params['model_prevents_outcome']
        self.drop_cols = sim_params['not_for_training']  # These columns are unusable for training

        # Store iteratively generated datasets here
        self.modified_datasets = dict()
        self.list_metrics = []

        # It's preferable to do it this way instead of modifying the Config class directly
        # Raises a KeyError in case something gets missed
        if run_params is None:
            run_params = Config.run_params_default

        # Redo the dataset according to configured division / downsampling / shuffling
        original_shape = self.dataset.shape[0]
        self.dataset = HelperFunctions.divide_dataset(
            self.dataset, self.time_var, self.outcome, self.dataset_div,
            downsample=run_params['downsample_dataset'],
            shuffle=run_params['shuffle_admissions'],
            random_state=Config.random_state)

        # Create a central "optimize" dictionary containing the random state
        # NOTE: Use a copy.copy() of this when sending it to the helper function
        self.optimize = None
        if run_params['hyperparameter_optimization']:
            self.optimize = {'random_state': Config.random_state}

        # Housekeeping
        # Print run parameters
        # NOTE Downsampling takes place prior to shuffling
        self.parallel = parallel
        if not self.parallel:
            if run_params['downsample_dataset']:
                print(f'Downsampled from {original_shape} to {self.dataset.shape[0]} samples')
            else:
                print('Downsampling: False')

            print('Model name:', run_params['model_name'])
            print('Admission shuffling:', run_params['shuffle_admissions'])
            print('Neglect conversions:', run_params['conversions_due_to_neglect'], Config.neglect_cap)
            print('Threshold calibration:', run_params['threshold_calibration'])
            print('Mitigation:', run_params['mitigation_strategy'])
            print('Hyperparameter Optimization:', run_params['hyperparameter_optimization'])

        # Assign run params
        self.run_params = run_params

        # Save results here
        dir_results = os.path.join(Config.dir_results, simulation_name)
        os.makedirs(dir_results, exist_ok=True)

        # Create a filename for the results
        outfile_name = '_'.join([
            simulation_name,
            run_params['model_name'],
            f'Downsampled{run_params["downsample_dataset"]}',
            f'NegInvert{run_params["conversions_due_to_neglect"]}',
            f'Threshold{run_params["threshold_calibration"]}',
            f'ShuffleAdmissions{run_params["shuffle_admissions"]}',
            f'Mitigation{run_params["mitigation_strategy"]}',
            f'HyperparamOpt{run_params["hyperparameter_optimization"]}'
        ])
        self.outfile_name = os.path.join(dir_results, outfile_name + '.pickle')

    @staticmethod
    def assign_mitigation_column(df):
        # Assign a mitigation column to the dataset
        if 'PREDICTIONS_WERE_IMPLEMENTED' not in df.columns:
            df = df.assign(PREDICTIONS_WERE_IMPLEMENTED=0)

        # Required: Will return a None if the condition isn't met
        return df

    def retrain(self, df_train, df_label, df_test, interval, effect_size):
        # Create an extra feature depending on mitigation strategy if it doesn't already exist
        if self.run_params['mitigation_strategy'] == 'FEATURE':
            df_train = self.assign_mitigation_column(df_train)
            df_label = self.assign_mitigation_column(df_label)
            df_test = self.assign_mitigation_column(df_test)

        # Training and label generation
        oc_dict = HelperFunctions.train_model(
            df_train.drop(self.drop_cols, axis=1),  # Will be subject to a train-test split
            [
                # Both of these intervals form the prospective test set together
                df_label.drop(self.drop_cols, axis=1),  # df_label is the next interval - will have labels inverted
                df_test.drop(self.drop_cols, axis=1)  # df_test is the interval after
            ],
            oc=self.outcome,
            model_name=self.run_params['model_name'],
            optimize=copy.copy(self.optimize),
            threshold_calibration=self.run_params['threshold_calibration'],
            random_state=Config.random_state
        )

        # Unpack OC dict
        y_all = oc_dict['y']
        pred_all = oc_dict['pred']
        threshold = oc_dict['threshold']
        current_perf = oc_dict['perf_metrics']

        # Unpack y_all and pred_all - Same order as provided above
        y_df_label, y_df_test = y_all
        pred_df_label, pred_df_test = pred_all

        # Calculate prospective test set performance
        df_prospective = pd.concat((
            pd.DataFrame({'TRUE': y_df_test, 'PRED': pred_df_test}),
            pd.DataFrame({'TRUE': y_df_label, 'PRED': pred_df_label})
        ))

        prospective_perf = HelperFunctions.eval_metrics(
            df_prospective['TRUE'], df_prospective['PRED'],
            threshold=threshold, bootstrap=True)

        # Now that we have the prospective performance metrics - we look at label inversions
        # Inversions will only happen for the predictions in df_label - since that's the next depoloyment interval
        cat_pred = (pred_df_label >= threshold).astype('int')  # Categorical predictions. Not dogs and that kind of thing.

        # CRITICAL Depending on effect size
        # Invert ground truth in true positives"""

        df_pred = pd.DataFrame((y_df_label, cat_pred, pred_df_label)).T
        df_pred.index = df_label.index
        df_pred.columns = ['TRUE', 'PRED', 'PROB']  # Probability is required for risk based inversions

        # LABEL INVERSIONS
        # True positives
        df_tp = df_pred.query('TRUE == 1 and PRED == 1')
        df_tp = df_tp.sample(
            frac=effect_size,
            random_state=Config.random_state)
        df_tp = df_tp.assign(TRUE=np.logical_not(df_tp['TRUE']).astype('int'))  # Invert ground truth

        # MITIGATIONS
        # First, build the mitigation dataframe
        df_mitigate = None
        if self.run_params['mitigation_strategy'] is not None:
            resample_frac = np.clip(effect_size * Config.additional_positives_scale_factor, 0, 1)

            # Mitigations for True Positives
            df_tp_mitigate = df_pred.query('TRUE == 1 and PRED == 1')
            n_mitigation_samples = df_tp_mitigate.sample(
                frac=resample_frac,
                random_state=Config.random_state).shape[0]
            additional_samples = n_mitigation_samples - df_tp.shape[0]

            # Overwrite df_tp_mitigate for the new number of samples
            df_tp_mitigate = df_pred.query('TRUE == 1 and PRED == 1')
            df_tp_mitigate = df_tp_mitigate.drop(df_tp.index)  # Make sure no duplicates are added
            df_tp_mitigate = df_tp_mitigate.sample(
                n=additional_samples,  # This is ONLY additional samples
                random_state=Config.random_state)

            # Mitigations for False Positives
            df_fp = df_pred.query('TRUE == 0 and PRED == 1')
            df_fp_mitigate = df_fp.sample(
                frac=resample_frac,
                random_state=Config.random_state)

            # Concatenate mitigation dataframes
            df_mitigate = pd.concat((df_tp_mitigate, df_tp, df_fp_mitigate))

        # Account for neglect leading to a secondary event cascade
        df_tn = pd.DataFrame()
        if self.run_params['conversions_due_to_neglect']:

            # Calculate a separate effect size for neglect
            # Sample frac signifies that number
            try:
                sample_frac = Config.neglect_cap[effect_size]
            except KeyError:
                sample_frac = Config.neglect_cap['default']

            df_tn = df_pred.query('TRUE == 0 and PRED == 0')
            df_tn = df_tn.sample(
                frac=sample_frac,
                random_state=Config.random_state)
            df_tn = df_tn.assign(TRUE=np.logical_not(df_tn['TRUE']).astype('int'))  # Invert ground truth

        df_invert = pd.concat((df_tn, df_tp))
        n_inverted_labels = {'TP': df_tp.shape[0], 'TN': df_tn.shape[0]}

        # Sanity checks - Here and below
        # NOTE flip_hamming goes to np.nan in case the model has no true positive or true negative predictions
        # essentially, it misclassifies everything
        flip_hamming = distance.hamming(
            df_invert['TRUE'].values,
            df_label.loc[df_invert.index, self.outcome].values)

        # Invert labels according to above shenanigans
        # Not sure why but even with .loc it's throwing warnings
        df_label.loc[df_invert.index, self.outcome] = df_invert['TRUE']

        if effect_size > 0 and not np.isnan(flip_hamming):
            try:
                assert flip_hamming == 1, 'Flip hamming distance is not 1'
                assert all((df_label.loc[df_invert.index, self.outcome] == df_invert['TRUE']).values)
            except AssertionError as e:
                df_error = pd.DataFrame({
                    'FILENAME': [self.outfile_name],
                    'EFFECT_SIZE': [effect_size],
                    'INTERVAL': [interval],
                    **self.run_params
                })
                df_error.to_csv(Config.error_file, mode='a', header=False)
                print('AssertionError', self.run_params)

        # Create concatenated MODIFIED dataset
        df_train_mod = pd.concat((df_train, df_label), sort=True)

        # Implement mitigation according to run params
        mitigation_strategy = self.run_params['mitigation_strategy']
        if mitigation_strategy is not None:
            # Creates an additional feature
            if mitigation_strategy == 'FEATURE':
                df_train_mod.loc[df_mitigate.index, 'PREDICTIONS_WERE_IMPLEMENTED'] = 1

            # Drops patients for whom a prediction has been implemented
            elif mitigation_strategy == 'DROP':
                df_train_mod = df_train_mod.drop(df_mitigate.index)

            else:
                raise ValueError('Invalid mitigation_strategy')

        # Save this for the next iteration
        self.modified_datasets[effect_size] = df_train_mod

        # Record predictions and labels for df_label
        y_and_pred = [y_df_label, pred_df_label]

        # Record metrics
        self.list_metrics.append((
            interval, df_train_mod.shape[0],
            df_test.shape[0], effect_size,
            current_perf, prospective_perf,
            n_inverted_labels, y_and_pred
        ))

    def interval_iterations(self):
        # Intervals start at 1
        all_intervals = sorted(self.dataset['INTERVAL'].dropna().unique())

        if self.parallel or self.run_params['hyperparameter_optimization']:
            disable_progressbar = True
        else:
            disable_progressbar = False
            print(f'Dividing into {len(all_intervals)} intervals')

        # We need to end at the 2nd last interval
        for interval in tqdm.tqdm(all_intervals[:-1], disable=disable_progressbar):
            if self.run_params['hyperparameter_optimization']:
                print('Interval', interval)
            next_interval = interval + 1

            for effect_size in Config.effect_sizes:
                # First interval - There's no modification of the base dataset yet
                if effect_size not in self.modified_datasets.keys():
                    df_train = self.dataset.query('INTERVAL <= @interval')

                else:  # Second iteration onwards
                    df_train = self.modified_datasets[effect_size]

                # Gen new labels for this
                df_label = self.dataset.query('INTERVAL == @next_interval')

                # Test on all remaining data
                df_test = self.dataset.query('INTERVAL > @next_interval')

                # Train for each effect size with a continuously updated dataset
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.retrain(df_train, df_label, df_test, interval, effect_size)

    def gen_perf_metrics(self):
        df_metrics = pd.DataFrame(
            self.list_metrics,
            columns=[
                'INTERVAL', 'TRAINING_SAMPLES',
                'TESTING_SAMPLES', 'EFFECT_SIZE',
                'CURRENT_PERF', 'PROSPECTIVE_PERF',
                'N_INVERTED', 'Y_AND_PRED'])

        # Expand columns
        df_perf = pd.DataFrame.from_records(df_metrics['CURRENT_PERF'])
        df_perf.columns = ['PERF_' + i.upper() for i in df_perf.columns]

        # Expand columns
        df_pr_perf = pd.DataFrame.from_records(df_metrics['PROSPECTIVE_PERF'])
        df_pr_perf.columns = ['PROS_PERF_' + i.upper() for i in df_pr_perf.columns]

        df_metrics = pd.concat((
            df_metrics.drop(['CURRENT_PERF', 'PROSPECTIVE_PERF'], axis=1),
            df_perf, df_pr_perf), axis=1)
        df_metrics = df_metrics.set_index(
            ['EFFECT_SIZE', 'INTERVAL']).sort_index()

        df_metrics.to_pickle(self.outfile_name)

    def hammer_time(self):
        # See if the sim is valid
        if not is_sim_valid(self.run_params):
            raise ValueError('Simulation is not valid')

        if not os.path.exists(self.outfile_name):
            self.interval_iterations()
            self.gen_perf_metrics()
        else:
            print(f'{self.outfile_name} already exists. Skipping.')

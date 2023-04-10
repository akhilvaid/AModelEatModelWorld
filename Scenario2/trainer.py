# Implementation of one model happens prior to the development of another

import os
import sys
import copy
import datetime

import pandas as pd
import numpy as np

from config import Config
sys.path.append('../Utils/'); from helper_functions import HelperFunctions, is_sim_valid  # Just disgusting


class Trainer:
    def __init__(self, simulation_name, run_params=None):
        # CRITICAL Any patient specific identifiers must always remain within the index
        self.simulation_name = simulation_name
        assert '_' not in simulation_name, 'Simulation name cannot contain underscores'

        sim_params = Config.simulations[simulation_name]  # Load training params from the config
        self.dataset = pd.read_pickle(sim_params['dataset'])
        self.first_oc = sim_params['first_oc']  # The OC for which a model has been deployed for a while
        self.second_oc = sim_params['second_oc']  # The OC for which a new model will be deployed.
        self.time_col = sim_params['admit_col']  # Col corresponding to order of patient admissions
        self.td_col = sim_params['td_col']  # Timedelta between 2nd and 1st outcomes

        # It's preferable to do it this way instead of modifying the Config class directly
        # Raises a KeyError in case something gets missed
        if run_params is None:
            run_params = Config.run_params_default

        # Downsampling
        original_shape = self.dataset.shape[0]
        if run_params['downsample_dataset']:
            pos_samples = self.dataset.query(f'{self.second_oc} == 1')
            neg_samples = self.dataset.query(f'{self.second_oc} == 0'). \
                sample(pos_samples.shape[0], random_state=Config.random_state)

            self.dataset = pd.concat((pos_samples, neg_samples))

        # Create a central "optimize" dictionary containing the random state
        # NOTE: Use a copy.copy() of this when sending it to the helper function
        self.optimize = None
        if run_params['hyperparameter_optimization']:
            self.optimize = {'random_state': Config.random_state}

        # Housekeeping
        # Print run parameters
        if run_params['downsample_dataset']:
            print(f'Downsampled from {original_shape} to {self.dataset.shape[0]} samples')
        else:
            print('Downsampling: None')

        print('Model name:', run_params['model_name'])
        print('Neglect conversions:', run_params['conversions_due_to_neglect'], Config.neglect_cap)
        print('Threshold calibration:', run_params['threshold_calibration'])
        print('Mitigation:', run_params['mitigation_strategy'])
        print('Dataset Division:', run_params['first_oc_train_interval'])
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
            f'Mitigation{run_params["mitigation_strategy"]}',
            f'FirstTrain{int(run_params["first_oc_train_interval"] * 100)}',
            f'HyperparamOpt{run_params["hyperparameter_optimization"]}'
        ])
        self.outfile_name = os.path.join(dir_results, outfile_name + '.pickle')

    def train(self):
        df = self.dataset.copy()

        # Create a train dataframe
        # Contains the first part of all admissions
        time_span = df[self.time_col].max() - df[self.time_col].min()
        initial_timedelta = time_span * self.run_params['first_oc_train_interval']
        train_limit = df[self.time_col].min() + initial_timedelta

        # Drop columns not required for training once the partitioning is complete
        drop_cols = [self.time_col, self.second_oc, self.td_col]

        # Initial (training) interval
        df_initial = df.query(f'{self.time_col} <= @train_limit')
        df_initial = df_initial.drop(drop_cols, axis=1)

        # Prospective deployment interval
        df_prospective = df.query(f'{self.time_col} > @train_limit')
        df_prospective_relevant = df_prospective.drop(drop_cols, axis=1)
        # print(df_initial.shape, df_prospective_relevant.shape)

        print('Samples in initial data split for model 1:', df_initial.shape[0])
        print('Samples in prospective data split for model 1:', df_prospective.shape[0])

        # Train and eval the model
        # For future reference, the threshold is calculated on another data split from the df_initial
        # The prospective data is only used for predictions
        oc1_dict = HelperFunctions.train_model(
            df_initial,
            predict_on=[df_prospective_relevant],  # Requires a list
            oc=self.first_oc,
            model_name=self.run_params['model_name'],
            optimize=copy.copy(self.optimize),
            threshold_calibration=self.run_params['threshold_calibration'],
            random_state=Config.random_state)

        # Unpack the return_dict
        y_prospective = oc1_dict['y']
        pred_prospective = oc1_dict['pred']
        threshold = oc1_dict['threshold']
        first_oc_current_metrics = oc1_dict['perf_metrics']

        # Get prospective performance of the first model
        # The prospective labels / predictions are returned as a list - important to get the first element
        first_oc_prospective_metrics = HelperFunctions.eval_metrics(
            y_prospective[0], pred_prospective[0], threshold,
            threshold_calibration=None,  # Only required if a threshold isn't provided
            bootstrap=True)

        # Create a dataframe with the predictions, as well as the outcomes
        # And the timediff col
        df_endpoint = pd.DataFrame((
            pred_prospective[0], y_prospective[0],
            df_prospective[self.second_oc].values,
            df_prospective[self.td_col].values
        )).T
        df_endpoint.index = df_prospective.index
        df_endpoint.columns = ['PRED_OC1', 'OC1', 'OC2', 'TD']

        # Calculate the effect size... effect size.
        all_metrics = []
        for effect_size in Config.effect_sizes:
            td_outer_limit = datetime.timedelta(days=Config.oc_diff_time_limit)

            # These are essentially True Positives
            # Not considering OC inversion for OC1 anymore at this point
            df_tp = df_endpoint.query('PRED_OC1 >= @threshold and OC1 == 1')

            # Restrict further to patients who had OC2 within the outer TD limit
            df_tp = df_tp.query('OC2 == 1 and TD <= @td_outer_limit')

            # Sample these outcomes according to effect size
            # For mitigation: sample to a size above the effect size according to supplied threshold
            df_invert = df_tp.sample(
                frac=effect_size,
                random_state=Config.random_state)

            # MITIGATIONS
            # First, build the mitigation dataframe
            df_mitigate = None
            if self.run_params['mitigation_strategy'] is not None:
                resample_frac = np.clip(effect_size * Config.additional_positives_scale_factor, 0, 1)

                # Mitigations for True Positives
                # Similar to df_tp above - but DOES NOT take into account OC2
                df_tp_no_oc2 = df_endpoint.query('PRED_OC1 >= @threshold and OC1 == 1')
                n_mitigation_samples = df_tp_no_oc2.sample(
                    frac=resample_frac,
                    random_state=Config.random_state).shape[0]
                additional_samples = n_mitigation_samples - df_invert.shape[0]

                # Sample (additionally) from previously unseen TP samples
                df_tp_mitigate = df_tp_no_oc2.drop(df_invert.index)
                df_tp_mitigate = df_tp_mitigate.sample(
                    n=additional_samples,
                    random_state=Config.random_state)

                # Mitigations for False Positives
                df_fp = df_endpoint.query('PRED_OC1 >= @threshold and OC1 == 0')
                df_fp_mitigate = df_fp.sample(
                    frac=resample_frac,
                    random_state=Config.random_state)

                # Final mitigation dataframe
                df_mitigate = pd.concat((df_tp_mitigate, df_invert, df_fp_mitigate))

            # Account for neglect leading to a secondary event cascade
            if self.run_params['conversions_due_to_neglect']:
                # We need false negatives here: Model says no - but person gets OC1
                # NOT considering True Negatives since even if the patient died after
                # the death was likely not situated along the same biological cascade / pathway
                df_fn = df_endpoint.query('PRED_OC1 < @threshold and OC1 == 1')

                # Doesn't need an additional consideration of the TD between admission and OC2
                # Since the label and TD already consider that
                df_fn = df_fn.query('OC2 == 1 and TD <= @td_outer_limit')

                # Calculate a separate effect size for neglect
                # Sample frac signifies that number
                try:
                    sample_frac = Config.neglect_cap[effect_size]
                except KeyError:
                    sample_frac = Config.neglect_cap['default']

                # Sample these outcomes according to effect size
                # NOTE Can't sample by OC2 risk here since we don't
                # have an idea of how severe these patients are
                df_invert_n = df_fn.sample(
                    frac=sample_frac,
                    random_state=Config.random_state)

                # Concatenate these dataframes
                df_invert = pd.concat((df_invert, df_invert_n))

            # Holy opposite day, Batman
            df_invert['OC2'] = np.logical_not(df_invert['OC2'].values).astype('int')

            # Update the whole dataset with these labels
            # es: effect size
            df_data_es = self.dataset.copy()
            df_data_es = df_data_es.drop([self.time_col, self.first_oc, self.td_col], axis=1)
            df_data_es.loc[df_invert.index, self.second_oc] = df_invert['OC2']
            # print(df_data_es[self.second_oc].value_counts(normalize=True))

            # Implement mitigation according to Config.mitigation_strategy
            if self.run_params['mitigation_strategy'] is not None:
                # Creates an additional feature
                if self.run_params['mitigation_strategy'] == 'FEATURE':
                    df_data_es['PREDICTIONS_WERE_IMPLEMENTED'] = 0
                    df_data_es.loc[df_mitigate.index, 'PREDICTIONS_WERE_IMPLEMENTED'] = 1

                # Drops patients for whom a prediction has been implemented
                elif self.run_params['mitigation_strategy'] == 'DROP':
                    df_data_es = df_data_es.drop(df_mitigate.index)

                else:
                    raise ValueError('Invalid mitigation strategy')

            # Train and eval model on this dataset for OC2
            oc2_dict = HelperFunctions.train_model(
                df_data_es,
                predict_on=[],
                oc=self.second_oc,
                model_name=self.run_params['model_name'],
                optimize=copy.copy(self.optimize),
                threshold_calibration=self.run_params['threshold_calibration'],
                random_state=Config.random_state)

            # Show metrics
            # print(f'Effect size: {effect_size} | Labels inverted: {df_invert.shape[0]} | Dataset size: {df_data_es.shape[0]} | Perfs: {cross_val_metrics}')

            # Store metrics
            all_metrics.append({
                'EFFECT_SIZE': effect_size,
                'LABELS_INVERTED': df_invert.shape[0],
                'OC2_METRICS': oc2_dict['perf_metrics'],
                'OC2_PREDS': [oc2_dict['y_test'], oc2_dict['pred_test']],
            })

        # Collate results
        df_metrics = pd.DataFrame.from_records(all_metrics)
        dict_metrics = {
            'oc1_current': first_oc_current_metrics,
            'oc1_prospective': first_oc_prospective_metrics,
            'oc2': df_metrics
        }

        pd.to_pickle(dict_metrics, self.outfile_name)  # Saves a dictionary

    def hammer_time(self):
        # See if the sim is valid
        if not is_sim_valid(self.run_params):
            raise ValueError('Simulation is not valid')

        if not os.path.exists(self.outfile_name):
            self.train()
        else:
            print(f'{self.outfile_name} already exists. Skipping.')

# Simulates multiple models being deployed in parallel

import os
import sys
import copy
import multiprocessing

import tqdm
import pandas as pd
import numpy as np

from config import Config, PRNGSeeds
sys.path.append('../Utils/'); from helper_functions import HelperFunctions, is_sim_valid  # Just disgusting


class Trainer:
    def __init__(self, simulation_name, run_params=None):  # Parallel by default
        self.simulation_name = simulation_name
        assert '_' not in simulation_name, 'Simulation name cannot contain underscores'

        # Load training params from the config
        sim_params = Config.simulations[simulation_name]
        self.dataset = pd.read_pickle(sim_params['dataset'])
        self.outcomes = sim_params['outcomes'][:]  # For future reference - YOU'RE GETTING A POINTER WITHOUT THE SLICE
        self.time_col = sim_params['time_col']
        self.redundant_cols = sim_params['redundant_cols']

        # It's preferable to do it this way instead of modifying the Config class directly
        # Raises a KeyError in case something gets missed
        if run_params is None:
            run_params = Config.run_params_default

        # Create a central "optimize" dictionary containing the random state
        # NOTE: Use a copy.copy() of this when sending it to the helper function
        self.optimize = None
        if run_params['hyperparameter_optimization']:
            self.optimize = {'random_state': Config.random_state}

        # Housekeeping
        if run_params['downsample_dataset']:  # Will be downsampled for each outcome and then concatenated
            original_shape = self.dataset.shape[0]

            all_samples = []
            for outcome in self.outcomes:
                pos_samples = self.dataset.query(f'{outcome} == 1')
                neg_samples = self.dataset.query(f'{outcome} == 0').\
                    sample(pos_samples.shape[0], random_state=Config.random_state)
                all_samples.append(pd.concat([pos_samples, neg_samples]))
                breakpoint()

            self.dataset = pd.concat(all_samples)
            self.dataset = self.dataset.drop_duplicates()   # Required to prevent data leaks
                                                            # Will prevent perfect dataset balancing

            print(f'Downsampled from {original_shape} to {self.dataset.shape[0]} samples')

        print('Model name:', run_params['model_name'])
        print('Train time split:', run_params['first_oc_train_interval'])
        print('False positives render invalid:', run_params['false_positives_render_invalid'])
        print('Dummy models:', run_params['n_dummies'])
        print('Hyperparameter Optimization:', run_params['hyperparameter_optimization'])
        print('Threshold calibration:', run_params['threshold_calibration'])

        # Assign run params
        self.run_params = run_params

        # Save results here
        dir_results = os.path.join(Config.dir_results, simulation_name)
        os.makedirs(dir_results, exist_ok=True)

        # Doing this first to allow resuming
        # Create a filename for the results
        outfile_name = '_'.join([
            self.simulation_name,
            run_params['model_name'],
            f'Downsampled{run_params["downsample_dataset"]}',
            f'Threshold{run_params["threshold_calibration"]}',
            f'FirstTrain{int(run_params["first_oc_train_interval"] * 100)}',
            f'FalsePositivesRenderInvalid{run_params["false_positives_render_invalid"]}',
            f'Dummies{run_params["n_dummies"]}',
            f'HyperparamOpt{run_params["hyperparameter_optimization"]}',
        ])
        self.outfile_name = os.path.join(dir_results, outfile_name + '.pickle')

    def train(self):
        df = self.dataset.copy()

        # Create a train dataframe
        # Contains the first part of all admissions
        time_span = df[self.time_col].max() - df[self.time_col].min()
        initial_timedelta = time_span * self.run_params['first_oc_train_interval']
        train_limit = df[self.time_col].min() + initial_timedelta

        # Dataset split according to time limit
        df_initial = df.query(f'{self.time_col} <= @train_limit')  # Initial
        df_prospective = df.query(f'{self.time_col} > @train_limit')  # Prospective

        print('Samples in prospective data split:', df_prospective.shape[0])
        print('Samples in initial data split:', df_initial.shape[0])

        # For each outcome, train and eval the model
        predictions = {}
        for outcome in self.outcomes:
            print('Training model for:', outcome)

            drop_cols = [self.time_col, *[i for i in self.outcomes if i != outcome]]
            drop_cols.extend(self.redundant_cols)

            df_initial_relevant = df_initial.drop(drop_cols, axis=1)
            df_prospective_relevant = df_prospective.drop(drop_cols, axis=1)

            # df_intial_relevant is further split into train and test
            # y_prospective, pred_prospective, threshold, _
            oc_dict = HelperFunctions.train_model(
                df_initial_relevant,
                predict_on=[df_prospective_relevant],
                oc=outcome,
                model_name=self.run_params['model_name'],
                optimize=copy.copy(self.optimize),
                threshold_calibration=self.run_params['threshold_calibration'],
                random_state=Config.random_state)

            # TODO New code - confirm this
            y_prospective = oc_dict['y']
            pred_prospective = oc_dict['pred']
            threshold = oc_dict['threshold']

            # Get prospective performance of the model - Not essential (just for logging)
            # The prospective labels / predictions are returned as a list - important to get the first element
            # prospective_metrics = HelperFunctions.eval_metrics(
            #     y_prospective[0], pred_prospective[0], threshold,
            #     threshold_calibration=None,  # Only required if a threshold isn't provided
            #     bootstrap=True)

            # Save the predictions and threshold
            predictions[f'{outcome}_PRED'] = pred_prospective[0]
            predictions[f'{outcome}_THRESHOLD'] = threshold

        # Attach the predictions to the original dataframe wrt the threshold
        df_endpoint = df_prospective[self.outcomes]
        for outcome in self.outcomes:
            pred_binary = (predictions[f'{outcome}_PRED'] >= predictions[f'{outcome}_THRESHOLD']).astype('int')
            df_endpoint = df_endpoint.assign(**{f'{outcome}_PRED': pred_binary})

        # Additional dummy outcome(s)
        for n_dummy in range(self.run_params['n_dummies']):
            df_endpoint[f'dummy{n_dummy}'] = np.random.randint(0, 2, size=df_endpoint.shape[0])
            df_endpoint[f'dummy{n_dummy}_PRED'] = np.random.randint(0, 2, size=df_endpoint.shape[0])
            self.outcomes.append(f'dummy{n_dummy}')

        return df_endpoint

    def calculate_downstream_effects(self, s_pred, effect_size):
        # Create a new dataframe corresponding to predictions with the outcomes as the index
        df = pd.DataFrame((
            [s_pred[i] for i in self.outcomes],
            [s_pred[i + '_PRED'] for i in self.outcomes],
            [s_pred[i + '_PROB'] for i in self.outcomes])).T
        df.index = self.outcomes
        df.columns = ['TRUE', 'PRED', 'PROB']

        # Check interaction of each prediction with effect size
        # Restrict to those which exceed the effect size threshold
        df['IMPLEMENTED'] = df['PROB'] > (1 - effect_size)  # Negation is important

        # All predictions are valid unless rendered invalid by a prior prediction
        # This is true regardless of effect size - that cannot be reliably determined for a single prediction downstream
        df['VALID'] = True

        # Implement the highest usable probability first
        # This eliminates the need for an additional randomization for implementation
        df_prob = df.query('IMPLEMENTED == True').sort_values('PROB', ascending=False)

        # Get potentially invalid outcomes - select from those which were implemented
        invalid_outcomes = df_prob.index.to_list()
        for outcome in df_prob.index:
            # Iteration for an outcome means that it is valid
            invalid_outcomes.remove(outcome)

            # Process this prediction
            continue_loop = HelperFunctions.categorize_prediction(
                df_prob.loc[outcome, 'TRUE'],
                df_prob.loc[outcome, 'PRED'],
                false_positives_render_invalid=self.run_params['false_positives_render_invalid'])

            # Loop exits if the prediction is invalid
            if continue_loop is False:
                break

        # Invalidate the remaining predictions
        df.loc[df.index.isin(invalid_outcomes), 'VALID'] = False

        # Return a dataframe that contains validity state for each prediction
        return pd.DataFrame(df['VALID']).T

    def eval_outcomes(self, pointers):
        # Unpack pointer tuple
        df_pred_base, random_state = pointers

        # Just making sure
        df_pred = df_pred_base.copy()

        # Post hoc evaluation of how outcomes interact
        # Store results for this run config here:
        dict_results = dict({random_state: {}})

        # Assign each prediction a random probability of implementation
        # This is where the hamming distance randomization is introduced
        prng_instance = np.random.RandomState(random_state)  # Reproducibility
        for outcome in self.outcomes:
            prng_prob = prng_instance.random(df_pred.shape[0])
            df_pred[f'{outcome}_PROB'] = prng_prob

        # Go row by row and calculate the downstream effects of such probabilities
        for effect_size in Config.effect_sizes:
            if Config.debug:
                print(f'Currently running for {effect_size=} and {random_state=}')

            # tqdm.tqdm.pandas(desc=f'Effect size: {effect_size}')
            df_valid = df_pred.apply(
                lambda x: self.calculate_downstream_effects(x, effect_size),
                axis=1)

            saved_index = df_valid.index.copy()
            df_valid = pd.concat(df_valid.values)
            df_valid.index = saved_index

            # Join the valid predictions with the original dataframe
            df_results = df_pred.join(df_valid, how='inner', rsuffix='_VALID')

            # Save results
            dict_results[random_state][effect_size] = df_results

        # Return the results for collation across random states
        return dict_results

    def hammer_time(self):
        # Check if Sim is valid
        if not is_sim_valid(self.run_params):
            raise ValueError('Simulation is not valid')

        # Check if results already exist
        if os.path.exists(self.outfile_name):
            print(f'{self.outfile_name} already exists. Skipping.')
            return

        # Train and eval models for each outcome. Return predictions.
        df_pred = self.train()

        # Evaluate the outcomes
        # Each iteration occurs with a different random state
        random_states = PRNGSeeds.seeds[:Config.pdist_random_iterations]

        if Config.debug:
            print('DEBUGGING ENABLED')
            rs_results = []
            for random_state in random_states:
                rs_results.append(self.eval_outcomes((df_pred, random_state)))

        else:
            print('Starting outcome evaluation')
            # Multiprocessing for self.eval_outcomes with a progressbar
            random_state_iter = [(df_pred, random_state) for random_state in random_states]

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            rs_results = list(tqdm.tqdm(pool.imap_unordered(
                    self.eval_outcomes, random_state_iter), total=len(random_state_iter)))
            pool.close()
            pool.join()

        # Combine the results from each random state
        dict_results = {}
        for random_state_res in rs_results:
            dict_results.update(random_state_res)

        # Save the results
        pd.to_pickle(dict_results, self.outfile_name)

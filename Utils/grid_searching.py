# Randomized grid searches for ResultConfig hyperparameters

import gc
import os
import uuid
import copy
import shutil
import itertools
import multiprocessing.dummy as multiprocessing

import tqdm
import xgboost
import pandas as pd
import numpy as np

from sklearn import model_selection, metrics


class RGSConfig:
    n_jobs = 1
    cv_folds = 3
    cross_val = True


class RGSCV:
    def __init__(self, hyperparameters, n_iterations, X, y, random_state):
        self.hyperparameters = hyperparameters  # The hyperparameters as a dictionary: list of values
        self.n_iterations = n_iterations

        # Reconstitute dataframe and create a stratified k fold object
        df = pd.concat([X, pd.Series(y, index=X.index, name='OC')], axis=1)

        self.data = {'train': [], 'test': []}
        if RGSConfig.cross_val:  # Stratified k-fold
            skf = model_selection.StratifiedKFold(
                n_splits=RGSConfig.cv_folds,
                shuffle=True,
                random_state=random_state)

            for train_index, test_index in skf.split(df.drop('OC', axis=1), df['OC']):
                self.data['train'].append(df.iloc[train_index])
                self.data['test'].append(df.iloc[test_index])

        else:
            train, test = model_selection.train_test_split(
                df, test_size=0.2,
                random_state=random_state,
                stratify=df['OC'])

            self.data['train'].append(train)
            self.data['test'].append(test)

        # Create a multiprocessing queue
        self.queue = multiprocessing.Manager().Queue()

        # Create a temporary directory
        self.temp_dir = os.path.join('TempModels', str(uuid.uuid4()))
        os.makedirs(self.temp_dir, exist_ok=True)

        # Housekeeping
        np.random.seed(random_state)

    def run(self, gpu_id, test_hyperparams):
        additional_params = {
            'tree_method': 'gpu_hist',
            'gpu_id': gpu_id,
            'n_jobs': 1,
            'eval_metric': 'logloss'
        }

        # Create a copy of the hyperparameters to avoid shenanigans later
        test_hyperparams_copy = copy.deepcopy(test_hyperparams)
        num_boost_round = test_hyperparams_copy['n_estimators']  # Equivalent to n_estimators
        test_hyperparams_copy.pop('n_estimators')  # Remove n_estimators from test_hyperparams
        test_hyperparams_copy.update(additional_params)  # Update test_hyperparams with additional_params

        model_filenames = []
        model_base = os.path.join(self.temp_dir, str(uuid.uuid4()))

        for counter, df_train in enumerate(self.data['train']):

            # X and y
            X_train, y_train = df_train.drop('OC', axis=1), df_train['OC'].values.ravel()

            # Construct DMatrix
            dtrain = xgboost.DMatrix(X_train, label=y_train)

            # Fit the model
            booster = xgboost.train(
                test_hyperparams_copy,
                dtrain,
                num_boost_round=num_boost_round,
                verbose_eval=False)

            # Save the model
            model_path = model_base + '_' + str(counter) + '.model'
            booster.save_model(model_path)

            # Add the model to the list
            model_filenames.append(model_path)

            del booster
            gc.collect()

        self.queue.put((test_hyperparams, model_filenames))

    def hammer_time(self):
        # Create the parameter grid
        hyperparam_grid = []
        for run_params in (self.hyperparameters,):
            for values in itertools.product(*map(run_params.get, run_params.keys())):
                hyperparam_grid.append(dict(zip(run_params.keys(), values)))

        # Shuffle this grid
        np.random.shuffle(hyperparam_grid)

        # Restrict the grid to the first n_iterations and split by n_jobs
        hyperparam_grid = hyperparam_grid[:self.n_iterations]

        # Add default XGBoost hyperparameters
        default_hyperparams = {
            'n_estimators': 100,
            'learning_rate': 0.3,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'colsample_bytree': 1}
        hyperparam_grid.append(default_hyperparams)

        # One GPU / Several cores
        # UBER HACK
        for these_hyperparams in tqdm.tqdm(hyperparam_grid, desc='Hyperparameters'):
            try:
                process = multiprocessing.Process(
                    target=self.run,
                    args=(0, these_hyperparams))
                process.start()
                process.join()
            except Exception as e:
                print(e)
                continue

        # Get results from the queue
        results = []
        while not self.queue.empty():
            hyperparams, model_filenames = self.queue.get()

            # Load the models
            all_aurocs = []
            for count, filename in enumerate(model_filenames):
                model = xgboost.Booster(model_file=filename)

                # Evaluate the model
                df_test = self.data['test'][count]

                # Get the test set predictions
                X_test, y_test = df_test.drop('OC', axis=1), df_test['OC'].values.ravel()
                y_pred = model.inplace_predict(X_test, validate_features=False)

                # Calculate AUROC
                auroc = metrics.roc_auc_score(y_test, y_pred)
                all_aurocs.append(auroc)

            # Mean AUROC
            mean_auroc = np.mean(all_aurocs)
            results.append((hyperparams, mean_auroc))

        # Consolidate results as dataframe
        df_results = pd.DataFrame(results, columns=['HYPERPARAMS', 'AUROC'])
        best_hyperparams = df_results['HYPERPARAMS'].iloc[df_results['AUROC'].idxmax()]

        # Remove temp directory
        shutil.rmtree(self.temp_dir)

        return best_hyperparams

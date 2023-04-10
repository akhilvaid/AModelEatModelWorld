# Helper functions to assist with calculations in other sims

import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from matplotlib.lines import Line2D

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from grid_searching import RGSCV

# USE CAREFULLY
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


class HelperFunctions:
    # Accessory config
    bootstrap_iterations = 500

    @staticmethod
    def get_model(model_name, optimize=None):
        # If optimize arg is provided, assume hyperparam optimization is required
        optimized_params = {}
        if optimize is not None:
            optimized_params = HyperparameterOptimizer.optimize(
                model_name=model_name,
                X=optimize['X'], y=optimize['y'],
                random_state=optimize['random_state'])

        model = None
        if model_name == 'LASSO':
            # n_jobs will have no effect unless the problem is multi-class
            model = LogisticRegression(
                penalty='l1',
                max_iter=1000,
                solver='saga',
                **optimized_params)

        elif model_name == 'XGB':
            model = XGBClassifier(
                use_label_encoder=False,
                tree_method='gpu_hist',  # TODO Account for non-GPU machines
                gpu_id=HelperFunctions.randomize_gpu_id(),
                n_jobs=8,
                eval_metric='logloss',
                **optimized_params)

        elif model_name == 'LogisticRegression':
            model = LogisticRegression(
                penalty='none',
                max_iter=1000,
                solver='saga')

        return model

    @staticmethod
    def impute(df):
        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(df.values)
        imputed = pd.DataFrame(imputed, columns=df.columns, index=df.index)

        return imputed, imputer

    @staticmethod
    def scale(df):
        # Scale data - Gets and returns a dataframe
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.values)
        scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)

        return scaled, scaler

    @staticmethod
    def ci_calc(x, method='numpy'):
        if method == 'numpy':
            ci = np.percentile(x, [2.5, 97.5])
        else:
            ci = np.array(sms.DescrStatsW(x).tconfint_mean())

        return ci

    @staticmethod
    def threshold_calculation(y, pred, threshold_calibration):
        # ROC curve calculation is required for all thresholds
        fpr, tpr, thresh = metrics.roc_curve(y, pred, drop_intermediate=False)

        if threshold_calibration == 'YOUDEN':
            threshold_idx = np.argmax(tpr - fpr)
        elif threshold_calibration == '90SENS':
            threshold_idx = np.argmin(abs(tpr - .9))
        elif threshold_calibration == '90SPEC':
            threshold_idx = np.argmin(abs(1 - fpr - .9))  # 1 - fpr is the specificity
        else:
            raise ValueError('Threshold calibration method not recognized')

        # https://github.com/scikit-learn/scikit-learn/issues/3097
        threshold = thresh[threshold_idx]
        if threshold > 1:
            threshold = thresh[threshold_idx + 1]

        assert threshold < 1, 'Threshold is greater than 1'
        return threshold

    @staticmethod
    def eval_metrics(y, pred, threshold=None, threshold_calibration=None, bootstrap=False):
        if threshold is None:  # Calculate afresh for threshold dependent metrics
            threshold = HelperFunctions.threshold_calculation(
                y, pred,
                threshold_calibration=threshold_calibration)

        # Store metrics here
        dict_metrics = dict()

        # Cross val calculation also goes through here - does NOT require bootstrapping
        if bootstrap:
            # Store bootstrap metrics here
            dict_metrics = dict()
            for metric in ['auroc', 'auprc', 'sens', 'spec']:
                dict_metrics[metric] = []

            # Create a dataframe containing y and pred
            df_metrics = pd.DataFrame({'y': y, 'pred': pred})

            # Do bootstrap resampling to get confidence intervals
            for _ in range(HelperFunctions.bootstrap_iterations):
                df_resample = df_metrics.sample(frac=1, replace=True)
                y_resample = df_resample['y']
                pred_resample = df_resample['pred']

                # Calculate metrics
                auroc = metrics.roc_auc_score(y_resample, pred_resample)
                prec, recall, _ = metrics.precision_recall_curve(y_resample, pred_resample)
                auprc = metrics.auc(recall, prec)
                sens = metrics.recall_score(y_resample, (pred_resample > threshold).astype(int))
                spec = metrics.recall_score(y_resample, (pred_resample >= threshold).astype(int), pos_label=0)

                # Store
                dict_metrics['auroc'].append(auroc)
                dict_metrics['auprc'].append(auprc)
                dict_metrics['sens'].append(sens)
                dict_metrics['spec'].append(spec)

        else:
            # AUROC
            dict_metrics['auroc'] = metrics.roc_auc_score(y, pred)

            # AUPRC
            prec, recall, _ = metrics.precision_recall_curve(y, pred)
            dict_metrics['auprc'] = metrics.auc(recall, prec)

            # SENSITIVITY
            dict_metrics['sens'] = metrics.recall_score(y, (pred >= threshold).astype(int))

            # SPECIFICITY
            dict_metrics['spec'] = metrics.recall_score(y, (pred >= threshold).astype(int), pos_label=0)

        return dict_metrics

    @staticmethod
    def predict_eval(df, model_name, model_ref, oc, return_metrics, threshold_calibration, scaler, imputer):

        # X Y split
        X = df.drop(oc, axis=1)
        y = df[oc].values.ravel()

        # Imputation -> Scaling
        if model_name != 'XGB':
            X = pd.DataFrame(
                scaler.transform(imputer.transform(X)),
                columns=X.columns,
                index=X.index)

        # Predict
        pred = model_ref.predict_proba(X)[:, 1]

        threshold = None
        perf_metrics = None
        if return_metrics:
            threshold = HelperFunctions.threshold_calculation(
                y, pred, threshold_calibration=threshold_calibration)
            perf_metrics = HelperFunctions.eval_metrics(y, pred, threshold, bootstrap=True)
            # return threshold, perf_metrics

        # Return the y and predictions for plain old eval
        # If return_metrics is True, go for a (_, _, var, var)
        # If return_metrics is False, go for a (var, var, _, _)
        return y, pred, threshold, perf_metrics

    @staticmethod
    def train_model(
            df_train_all: pd.DataFrame, predict_on: list, oc: str, model_name: str,
            optimize, threshold_calibration=None, random_state=42):

        # Return things as part of this dictionary
        return_dict = dict()

        # Just in case
        df_train_all = df_train_all.copy()

        # Imputation -> Scaling
        scaler = imputer = None
        if model_name != 'XGB':
            df_X = df_train_all.drop(oc, axis=1)
            df_y = df_train_all[oc]

            # Imputation: Doesn't require separation of X and y
            df_X, imputer = HelperFunctions.impute(df_X)
            df_X, scaler = HelperFunctions.scale(df_X)

            df_train_all = pd.concat([df_X, df_y], axis=1)

        # Do a train-test split on the data
        df_train, df_test = train_test_split(
            df_train_all,
            test_size=0.2,
            random_state=random_state,
            stratify=df_train_all[oc])

        # Train a model on this dataframe the usual way
        X_train = df_train.drop(oc, axis=1)
        y_train = df_train[oc].values.ravel()

        # Hyperparameter optimization
        # If this is to be done, the "optimize" dict will arrive here containing the random_state
        if optimize is not None:  # TODO
            optimize['X'] = df_train_all.drop(oc, axis=1)
            optimize['y'] = df_train_all[oc].values.ravel()

        # Fit the model as usual
        model = HelperFunctions.get_model(model_name, optimize)
        model.fit(X_train, y_train)

        # We need perf on a train test split
        y_test, pred_test, threshold, perf_metrics = HelperFunctions.predict_eval(
            df_test,
            model_name=model_name, model_ref=model, oc=oc,
            return_metrics=True,
            threshold_calibration=threshold_calibration,
            scaler=scaler, imputer=imputer)

        # Add values to return dict
        return_dict['y_test'] = y_test
        return_dict['pred_test'] = pred_test
        return_dict['threshold'] = threshold
        return_dict['perf_metrics'] = perf_metrics

        # Predictions on the df_predict_on dataframe
        y_all = []
        pred_all = []
        for this_df in predict_on:

            # Eventually empty stuff will show up here
            # Retraining: df_test for the last interval
            if this_df.empty:
                y_all.append([])
                pred_all.append([])
                continue

            this_y, this_pred, _, _ = HelperFunctions.predict_eval(
                this_df,
                model_name=model_name, model_ref=model,
                oc=oc, return_metrics=False,
                threshold_calibration=None,
                scaler=scaler, imputer=imputer)

            y_all.append(this_y)
            pred_all.append(this_pred)

        # Add these to the return dict as well
        return_dict['y'] = y_all
        return_dict['pred'] = pred_all

        return return_dict

    @staticmethod
    def categorize_prediction(ground_truth, prediction, false_positives_render_invalid):
        # Accepts ground truth and prediction
        # Returns the prediction state, and whether to continue the prediction loop
        # Whether to continue the prediction loop affects validity of following predictions

        # Default: Continue the loop
        continue_prediction_loop = True

        # True positives
        if ground_truth == 1 and prediction == 1:
            continue_prediction_loop = False

        # False positives
        if ground_truth == 0 and prediction == 1:
            if false_positives_render_invalid:
                continue_prediction_loop = False

        # The prediction loop will continue for all negative predictions
        return continue_prediction_loop

    @staticmethod
    def divide_dataset(dataset, time_col, outcome, chunks, downsample, shuffle, random_state):
        # Division of dataset into temporal chunks
        min_time = dataset[time_col].min()
        max_time = dataset[time_col].max()

        diff = max_time - min_time
        partition_increment = diff / chunks  # Arbitrary divisions or those mandated by the simulation
        partitions = [
            min_time + (i * partition_increment)
            for i in range(chunks + 1)]
        s_interval = pd.cut(
            dataset[time_col],
            partitions, labels=range(1, chunks + 1))
        dataset = dataset.assign(INTERVAL=s_interval.fillna(1))

        # Assumes positive samples > negative samples
        if downsample:
            pos_samples = dataset.query(f'{outcome} == 1')
            neg_samples = dataset.query(f'{outcome} == 0'). \
                sample(pos_samples.shape[0], random_state=random_state)

            dataset = pd.concat((pos_samples, neg_samples))

        if shuffle:
            dataset[time_col] = dataset[time_col].\
                sample(frac=1, random_state=random_state).values  # Random state constant for reproducibility

        return dataset


class HyperparameterOptimizer:
    # This is a shortcut list
    # See earlier commits for the full list of hyperparameters
    parameters_xgb = {
        'n_estimators': [40, 60, 80],
        'learning_rate': [0.10, 0.20],
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.2, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
    }

    parameters_lasso = {
        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    }

    cv_folds = 3
    xgb_iter = 2000
    lasso_iter = len(parameters_lasso['C'])

    @classmethod
    def optimize(cls, model_name, X, y, random_state):
        # Hybrid method involving sklearn and my own f*ckery
        # Tailored to Minerva
        if model_name == 'XGB':
            rgscv = RGSCV(
                hyperparameters=cls.parameters_xgb,
                n_iterations=cls.xgb_iter,
                X=X,
                y=y,
                random_state=random_state)

            best_hyperparameters = rgscv.hammer_time()

        elif model_name == 'LASSO':
            model = LogisticRegression(
                penalty='l1',
                max_iter=1000,
                solver='saga')

            rgs = RandomizedSearchCV(
                model,
                cls.parameters_lasso,
                n_iter=cls.lasso_iter,
                scoring='roc_auc',
                n_jobs=-1,
                cv=cls.cv_folds,
                random_state=random_state,
                verbose=0)

            rgs.fit(X, y)
            best_hyperparameters = rgs.cv_results_['params'][rgs.best_index_]

        elif model_name == 'LogisticRegression':
            best_hyperparameters = None

        else:
            raise NotImplementedError(f'Model {model_name} not implemented')

        return best_hyperparameters


def is_sim_valid(run_params) -> bool:
    """
    Proceed with the sim only if certain conditions are met
    :param run_params:
    :return:
    """

    # Logistic Regression can't have hyperparameter optimization
    if run_params['model_name'] == 'LogisticRegression':
        try:
            if run_params['hyperparameter_optimization'] == 'True':
                return False
        except KeyError:
            pass

    return True

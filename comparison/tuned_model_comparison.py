from typing import Dict

import numpy as np
import optuna
import pandas as pd
import time
from optuna import Trial
from optuna.integration.sklearn import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold
from comparison.comparison_datasets import ComparisonDataset
from comparison.model_comparison import ModelComparison, ModelName, MODEL_SCORE, TRAINING_TIME
from comparison.tuning_parameters import TuningParameters

NUM_TRIALS = "num_trials"
BEST_PARAMETERS = "best_parameters"


class TunedModelComparison(ModelComparison):

    def __init__(self,
                 comparison_dataset: ComparisonDataset,
                 max_parameters_to_test_in_tuning: int = 300,
                 early_stopping_patience: int = 100,
                 early_stopping_min_delta: float = 0.005):
        self.max_parameters_to_test_in_tuning = max_parameters_to_test_in_tuning
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        super().__init__(comparison_dataset)

    def _get_default_model_score_and_training_time(self, model_name: ModelName) -> Dict[str, object]:
        start_time = time.time()
        model = self.models_to_compare[model_name][self.task_name]
        best_score_and_params = self._tune_model(model, model_name)
        end_time = time.time()
        return {TRAINING_TIME: end_time - start_time, **best_score_and_params}

    def _tune_model(self, model: BaseEstimator, model_name: ModelName) -> Dict[str, object]:
        def objective(trial: Trial) -> float:
            grid_params: Dict = TuningParameters().get_model_params(model_name)(trial)
            model.set_params(**grid_params)
            cross_val_scores = cross_val_score(model, self.preprocessed_features, self.target, n_jobs=4,
                                               cv=KFold(self.cross_validation_n_folds, shuffle=True))
            if pd.isnull(cross_val_scores).any():
                return - np.inf
            return np.mean(cross_val_scores)

        study = optuna.create_study(direction="maximize")
        try:
            study.optimize(objective,
                           n_trials=self.max_parameters_to_test_in_tuning,
                           callbacks=[self._early_stopping_callback])
        except Exception as e:
            print(f"Exception while tuning model {model_name}: {e}")
        return {MODEL_SCORE: study.best_value,
                BEST_PARAMETERS: str(study.best_params),
                NUM_TRIALS: len(study.trials)}

    def _early_stopping_callback(self, study, _):
        if EarlyStoppingExceeded.best_score is None:
            EarlyStoppingExceeded.best_score = study.best_value

        if study.best_value > EarlyStoppingExceeded.best_score + self.early_stopping_min_delta:
            EarlyStoppingExceeded.best_score = study.best_value
            EarlyStoppingExceeded.early_stop_count = 0
        else:
            if EarlyStoppingExceeded.early_stop_count > self.early_stopping_patience:
                EarlyStoppingExceeded.early_stop_count = 0
                raise EarlyStoppingExceeded()
            else:
                EarlyStoppingExceeded.early_stop_count = EarlyStoppingExceeded.early_stop_count + 1


class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop_count = 0
    best_score = None

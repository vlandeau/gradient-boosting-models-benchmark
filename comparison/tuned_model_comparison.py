import time
from typing import Tuple, Dict

import numpy as np
import optuna
from optuna import Trial
from optuna.integration.sklearn import BaseEstimator
from sklearn.model_selection import cross_val_score

from comparison.model_comparison import ModelComparison, ModelName
from comparison.tuning_parameters import TuningParameters


class TunedModelComparison(ModelComparison):

    def _get_default_model_score_and_training_time(self, model_name: ModelName) -> Tuple[float, float]:
        start_time = time.time()
        model = self.models_to_compare[model_name][self.task_name]
        best_score = self._tune_model(model, model_name)
        end_time = time.time()
        return best_score, end_time - start_time

    def _tune_model(self, model: BaseEstimator, model_name: ModelName) -> float:
        def objective(trial: Trial) -> float:
            grid_params: Dict = TuningParameters().get_model_params(model_name)(trial)
            model.set_params(**grid_params)
            return np.mean(cross_val_score(model, self.features, self.target,
                                           cv=self.cross_validation_n_folds))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.max_parameters_to_test_in_tuning)

        return study.best_value

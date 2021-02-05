import time
from typing import Dict

import numpy as np
import optuna
from optuna import Trial
from optuna.integration.sklearn import BaseEstimator
from sklearn.model_selection import cross_val_score

from comparison.model_comparison import ModelComparison, ModelName, MODEL_SCORE, TRAINING_TIME
from comparison.tuning_parameters import TuningParameters

BEST_PARAMETERS = "best_parameters"


class TunedModelComparison(ModelComparison):

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
            return np.mean(cross_val_score(model, self.preprocessed_features, self.target,
                                           cv=self.cross_validation_n_folds))

        study = optuna.create_study(direction="maximize")
        try:
            study.optimize(objective, n_trials=self.max_parameters_to_test_in_tuning)
            return {MODEL_SCORE: study.best_value,
                    BEST_PARAMETERS: str(study.best_params)}
        except Exception as e:
            print(f"Exception while tuning model {model_name}: {e}")
            return {MODEL_SCORE: np.nan}

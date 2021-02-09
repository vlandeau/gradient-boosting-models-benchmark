from typing import Dict, Callable, Union

from optuna import Trial

from comparison.model_comparison import ModelName


class TuningParameters:

    def get_model_params(self, model_name: ModelName) -> Callable[[Trial], Dict[str, Union[int, float, str]]]:
        return {
            ModelName.CATBOOST: self.get_catboost_params(f"{ModelName.CATBOOST.value}__"),
            ModelName.XGBOOST: self.get_xgboost_params(f"{ModelName.XGBOOST.value}__"),
            ModelName.LIGHTGBM: self.get_lightgbm_params(f"{ModelName.LIGHTGBM.value}__"),
            ModelName.LIGHTGBM_WITH_CATBOOST_ENCODER: self.get_lightgbm_params(f"{ModelName.LIGHTGBM.value}__")
        }[model_name]

    @staticmethod
    def get_lightgbm_params(model_step_in_pipeline: str = "") -> Callable[[Trial], Dict[str, Union[int, float, str]]]:
        return lambda trial: {
            f"{model_step_in_pipeline}boosting_type": trial.suggest_categorical('boosting_type', ["gbdt", "dart", "goss"]),
            f"{model_step_in_pipeline}num_leaves": trial.suggest_int('num_leaves', 30, 150),
            f"{model_step_in_pipeline}learning_rate": trial.suggest_float('learning_rate', 0.001, 0.2),
            f"{model_step_in_pipeline}subsample_for_bin": trial.suggest_int('subsample_for_bin', 20000, 300000),
            f"{model_step_in_pipeline}feature_fraction": trial.suggest_float('feature_fraction', 0., 1.),
            f"{model_step_in_pipeline}bagging_fraction": trial.suggest_float('bagging_fraction', 0., 1.),
            f"{model_step_in_pipeline}min_data_in_leaf": trial.suggest_int('min_data_in_leaf', 0, 100),
            f"{model_step_in_pipeline}lambda_l1": trial.suggest_float('lambda_l1', 0, 10),
            f"{model_step_in_pipeline}lambda_l2": trial.suggest_float('lambda_l2', 0, 10)
        }

    @staticmethod
    def get_catboost_params(model_step_in_pipeline: str = "") -> Callable[[Trial], Dict[str, Union[int, float, str]]]:
        return lambda trial: {
            f"{model_step_in_pipeline}depth": trial.suggest_int("depth", 1, 16),
            f"{model_step_in_pipeline}n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            f"{model_step_in_pipeline}bagging_temperature": trial.suggest_float("bagging_temperature", 1, 100,
                                                                                log=True),
            f"{model_step_in_pipeline}learning_rate": trial.suggest_float('learning_rate', 0.001, 0.2),
            f"{model_step_in_pipeline}l2_leaf_reg": trial.suggest_float('l2_leaf_reg', 0, 10)
        }

    @staticmethod
    def get_xgboost_params(model_step_in_pipeline: str = "") -> Callable[[Trial], Dict[str, Union[int, float, str]]]:
        return lambda trial: {
            f"{model_step_in_pipeline}max_depth" : trial.suggest_int("max_depth", 2, 30),
            f"{model_step_in_pipeline}booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            f"{model_step_in_pipeline}learning_rate" : trial.suggest_float("learning_rate", 0.001, 0.5),
            f"{model_step_in_pipeline}n_estimators" : trial.suggest_int("n_estimators", 20, 205),
            f"{model_step_in_pipeline}gamma" : trial.suggest_float("gamma", 0, 0.5),
            f"{model_step_in_pipeline}min_child_weight" : trial.suggest_int("min_child_weight", 0, 10),
            f"{model_step_in_pipeline}subsample" : trial.suggest_float("subsample", 0.1, 1),
            f"{model_step_in_pipeline}colsample_bytree" : trial.suggest_float("colsample_bytree", 0.1, 1.0)
        }
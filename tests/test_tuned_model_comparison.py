import pandas as pd
import numpy as np
from hamcrest import assert_that

from comparison.comparison_datasets import ComparisonDataset, TaskName
from comparison.model_comparison import MODEL_SCORE
from comparison.tuned_model_comparison import TunedModelComparison

n_samples = 20
numerical_target = pd.Series(np.random.normal(size=n_samples))


def test_model_comparison_give_non_null_performance_with_regression_and_numerial_feature():
    # Given
    cross_validation_n_folds = 2
    features = pd.DataFrame({
        "numeric_feature": np.random.normal(size=n_samples)
    })
    comparison_dataset = ComparisonDataset(TaskName.REGRESSION, features, numerical_target, cross_validation_n_folds)

    # When
    comparison = TunedModelComparison(comparison_dataset).get_default_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        performance = performance_and_training_time[MODEL_SCORE]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")


def test_model_comparison_give_non_null_performance_with_regression_and_categorical_feature():
    # Given
    cross_validation_n_folds = 2
    features = pd.DataFrame({
        "string_feature": list(np.random.choice(["Paris", "London", "Madrid", "Roma"], n_samples - 1)) + [None],
    })
    comparison_dataset = ComparisonDataset(TaskName.REGRESSION, features, numerical_target, cross_validation_n_folds)

    # When
    comparison = TunedModelComparison(comparison_dataset).get_default_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        performance = performance_and_training_time[MODEL_SCORE]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")

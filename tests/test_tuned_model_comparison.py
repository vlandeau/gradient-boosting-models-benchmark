import pandas as pd
import numpy as np
from hamcrest import assert_that
from hamcrest.core.core import is_

from comparison.model_comparison import TaskName
from comparison.tuned_model_comparison import TunedModelComparison


def test_model_comparison_give_non_null_performance_with_regression():
    # Given
    n_samples = 20
    cross_validation_n_folds = 2
    numerical_target = pd.Series(np.random.normal(size=n_samples))
    features = pd.DataFrame({
        "numeric_feature": np.random.normal(size=n_samples)
    })

    # When
    model_comparison = TunedModelComparison(TaskName.regression, cross_validation_n_folds, features, numerical_target)
    comparison = model_comparison.get_default_models_scores_and_training_time()

    # Then
    for model_name, performance_and_training_time in comparison.items():
        assert_that(len(performance_and_training_time), is_(2),
                    reason=f"Insufficiant metric count for model {model_name}")
        performance = performance_and_training_time[0]
        training_time = performance_and_training_time[1]
        assert_that(~np.isnan(performance),
                    reason=f"Null performance value for model {model_name}")

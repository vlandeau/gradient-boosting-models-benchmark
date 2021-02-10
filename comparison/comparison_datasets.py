from dataclasses import dataclass
from enum import Enum
from typing import Dict

from pandas import DataFrame, Series, read_csv
from sklearn import datasets


class TaskName(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ComparisonDataset():
    task: TaskName
    features: DataFrame
    target: Series
    cross_validation_n_folds: int


def get_comparison_datasets() -> Dict[str, ComparisonDataset]:
    datasets_information = {}

    california_housing = datasets.fetch_california_housing(as_frame=True)
    datasets_information['california'] = ComparisonDataset(TaskName.REGRESSION,
                                                           california_housing.data,
                                                           california_housing.target,
                                                           4)

    covtype_features, covtype_target = datasets.fetch_covtype(return_X_y=True)
    datasets_information['covtype'] = ComparisonDataset(TaskName.CLASSIFICATION,
                                                        DataFrame(covtype_features),
                                                        Series(covtype_target),
                                                        2)
    adult = datasets.fetch_openml('adult', as_frame=True)
    datasets_information['adult'] = ComparisonDataset(TaskName.CLASSIFICATION,
                                                      adult.data,
                                                      adult.target,
                                                      4)
    ukair = datasets.fetch_openml('particulate-matter-ukair-2017', as_frame=True)
    datasets_information['ukair'] = ComparisonDataset(TaskName.REGRESSION,
                                                      ukair.data,
                                                      ukair.target,
                                                      2)
    diabetes = datasets.fetch_openml('diabetes', as_frame=True)
    datasets_information['diabetes'] = ComparisonDataset(TaskName.CLASSIFICATION,
                                                         diabetes.data,
                                                         diabetes.target,
                                                         10)
    bank_marketing = datasets.fetch_openml("bank-marketing", as_frame=True)
    datasets_information['bank'] = ComparisonDataset(TaskName.CLASSIFICATION,
                                                     bank_marketing.data,
                                                     bank_marketing.target,
                                                     4)
    speed_dating = datasets.fetch_openml("SpeedDating", as_frame=True)
    datasets_information['dating'] = ComparisonDataset(TaskName.CLASSIFICATION,
                                                       speed_dating.data,
                                                       speed_dating.target,
                                                       6)
    hill_valley = datasets.fetch_openml("hill-valley", as_frame=True)
    datasets_information['valley'] = ComparisonDataset(TaskName.CLASSIFICATION,
                                                       hill_valley.data,
                                                       hill_valley.target,
                                                       8)
    cars: DataFrame = read_csv("cars.csv")
    cars_target = "duration_listed"
    datasets_information['cars'] = ComparisonDataset(TaskName.REGRESSION,
                                                     cars.drop(columns=cars_target),
                                                     cars[cars_target],
                                                     4)
    return datasets_information

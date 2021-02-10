from typing import Dict

import pandas as pd
from sklearn import datasets
from tqdm import tqdm
from model_comparison import ModelComparison, TaskName
from tuned_model_comparison import TunedModelComparison
import json


def get_comparison_datasets() -> Dict:
    datasets_information = {}

    california_housing = datasets.fetch_california_housing(as_frame=True)
    datasets_information['california'] = {"task": TaskName.REGRESSION,
                                          "features": california_housing.data,
                                          "target": california_housing.target,
                                          "cv": 4}
    # covtype_features, covtype_target = datasets.fetch_covtype(return_X_y=True)
    # datasets_information['covtype'] = {"task": TaskName.CLASSIFICATION,
    #                                    "features": pd.DataFrame(covtype_features),
    #                                    "target": pd.Series(covtype_target),
    #                                    "cv": 2}
    # adult = datasets.fetch_openml('adult', as_frame=True)
    # datasets_information['adult'] = {"task": TaskName.CLASSIFICATION,
    #                                  "features": adult.data,
    #                                  "target": adult.target,
    #                                  "cv": 4}
    # ukair = datasets.fetch_openml('particulate-matter-ukair-2017', as_frame=True)
    # datasets_information['ukair'] = {"task": TaskName.REGRESSION,
    #                                  "features": ukair.data,
    #                                  "target": ukair.target,
    #                                  "cv": 2}
    # diabetes = datasets.fetch_openml('diabetes', as_frame=True)
    # datasets_information['diabetes'] = {"task": TaskName.CLASSIFICATION,
    #                                     "features": diabetes.data,
    #                                     "target": diabetes.target,
    #                                     "cv": 10}
    # bank_marketing = datasets.fetch_openml("bank-marketing", as_frame=True)
    # datasets_information['bank'] = {"task": TaskName.CLASSIFICATION,
    #                                 "features": bank_marketing.data,
    #                                 "target": bank_marketing.target,
    #                                 "cv": 4}
    # speed_dating = datasets.fetch_openml("SpeedDating", as_frame=True)
    # datasets_information['dating'] = {"task": TaskName.CLASSIFICATION,
    #                                   "features": speed_dating.data,
    #                                   "target": speed_dating.target,
    #                                   "cv": 6}
    # hill_valley = datasets.fetch_openml("hill-valley", as_frame=True)
    # datasets_information['valley'] = {"task": TaskName.CLASSIFICATION,
    #                                   "features": hill_valley.data,
    #                                   "target": hill_valley.target,
    #                                   "cv": 8}
    # cars = pd.read_csv("cars.csv")
    # cars_target = "duration_listed"
    # datasets_information['cars'] = {"task": TaskName.REGRESSION,
    #                                 "features": cars.drop(columns=cars_target),
    #                                 "target": cars[cars_target],
    #                                 "cv": 4}
    return datasets_information


def get_comparison_default_models(dataset_infos, dataset_name):
    print(f"Processing {dataset_name} dataset")
    comparison = ModelComparison(task_name=dataset_infos["task"],
                                 cross_validation_n_folds=dataset_infos["cv"],
                                 features=dataset_infos["features"],
                                 target=dataset_infos["target"])
    return comparison.get_default_models_scores_and_training_time()


def get_comparison_tuned_models(dataset_infos, dataset_name):
    print(f"Processing {dataset_name} dataset")
    comparison = TunedModelComparison(task_name=dataset_infos["task"],
                                      cross_validation_n_folds=dataset_infos["cv"],
                                      features=dataset_infos["features"],
                                      target=dataset_infos["target"])
    return comparison.get_default_models_scores_and_training_time()


if __name__ == "__main__":
    comparison_datasets = get_comparison_datasets()

    perf_comparisons = {dataset_name: get_comparison_default_models(comparison_datasets[dataset_name], dataset_name)
                        for dataset_name in comparison_datasets.keys()}
    with open("perf_comparison.json", "w") as default_performances_output_stream:
        json.dump(perf_comparisons, default_performances_output_stream)

    tuned_perf_comparisons = {dataset_name: get_comparison_tuned_models(comparison_datasets[dataset_name], dataset_name)
                              for dataset_name in tqdm(comparison_datasets.keys())}
    with open("tuned_perf_comparisons.json", "w") as tuned_performances_output_stream:
        json.dump(perf_comparisons, tuned_performances_output_stream)

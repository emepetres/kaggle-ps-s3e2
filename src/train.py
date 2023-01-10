import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

import config
from model_dispatcher import (
    CustomModel,
    DecisionTreeModel,
    DecisionTreeModelSVD,
    XGBoost,
)  # , LightGBM, CatBoost


def run(fold: int, model: CustomModel) -> Tuple[float, np.ndarray]:
    # load the full training data with folds
    df = pd.read_csv(config.TRAIN_FOLDS)
    df_test = pd.read_csv(config.TEST_DATA)
    # # df_test["synthetic_data"] = 1

    # all columns are features except target, id and kfold columns
    features = [f for f in df.columns if f not in (config.TARGET, "kfold", "id")]
    ord_features = ["age", "avg_glucose_level", "bmi"]
    cat_features = [
        f for f in features if f not in ord_features
    ]  # all original features are ordinal

    # initialize model
    custom_model = model(
        df, fold, config.TARGET, cat_features, ord_features, test=df_test
    )

    # encode all features
    custom_model.encode()

    # fit model on training data
    custom_model.fit()

    # predict on validation data and get rmse score
    auc = custom_model.predict_and_score()

    # print rmse
    print(f"Fold = {fold}, AUC = {auc}")

    # predict on test
    preds = custom_model.predict_test()

    return (auc, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="xgb")

    args = parser.parse_args()

    model = None
    if args.model == "rf":
        model = DecisionTreeModel
    elif args.model == "svd":
        model = DecisionTreeModelSVD
    elif args.model == "xgb":
        model = XGBoost
    # # elif args.model == "lgbm":
    # #     model = LightGBM
    # # elif args.model == "cb":
    # #     model = CatBoost
    else:
        raise argparse.ArgumentError(
            "Only 'rf' (random forest)"
            ", 'svd' (random forest with truncate svd)"
            ", 'xgb' (XGBoost)"
            # # ", 'lgbm (LightGBM)'"
            # # ", 'cb' (CatBoost)"
            # # "and 'emb' (Tabular NN with embeddings)"
            " models are supported"
        )

    validation_scores = []
    preds = []
    for fold_ in range(config.FOLDS):
        score, predictions = run(fold_, model)
        validation_scores.append(score)
        preds.append(predictions)

    valid_auc = np.mean(validation_scores)
    print(f"Validation AUC = {valid_auc}")

    # create submission
    pred = np.mean(np.array(preds), axis=0)

    df_sub = pd.read_csv(config.SUBMISSION_SAMPLE)
    df_sub[config.TARGET] = pred

    dt = datetime.now().strftime("%y%m%d.%H%M")
    submission_file = Path(config.OUTPUTS) / f"{dt}-{args.model}-{valid_auc}.csv"
    submission_file.parent.mkdir(exist_ok=True)
    df_sub.to_csv(submission_file, index=False)

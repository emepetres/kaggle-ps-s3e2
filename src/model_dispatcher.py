from typing import List
import pandas as pd
import numpy as np

from sklearn import ensemble, metrics
import xgboost as xgb

from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgbm
# # from catboost import CatBoostRegressor

from common.encoding import (
    reduce_dimensions_svd,
    encode_to_values,
)


class CustomModel:
    def __init__(
        self,
        data: pd.DataFrame,
        fold: int,
        target: str,
        cat_features: List[str],
        ord_features: List[str],
        test: pd.DataFrame = None,
    ):
        self.data = data
        self.fold = fold
        self.target = target
        self.cat_features = cat_features
        self.ord_features = ord_features
        self.test = test

        self.features = cat_features + ord_features

    def encode(self):
        """Transforms data into x_train & x_valid"""
        pass

    def fit(self):
        """Fits the model on x_valid and train target"""
        pass

    def predict_and_score(self) -> float:
        """Predicts on x_valid data and score using AUC"""
        # we need the probability values as we are calculating RMSE
        # we will use the probability of 1s
        valid_preds = self.model.predict_proba(self.x_valid)[:, 1]

        return metrics.roc_auc_score(self.df_valid[self.target].values, valid_preds)

    def predict_test(self) -> np.ndarray:
        """Predicts on x_test data"""

        if self.test is None:
            return None

        # we will use the probability of 1s
        return self.model.predict_proba(self.x_test)[:, 1]


class DecisionTreeModel(CustomModel):
    def encode(self):
        encode_to_values(self.data, self.cat_features, test=self.test)

        # get training & validation data using folds
        self.df_train = self.data[self.data.kfold != self.fold].reset_index(drop=True)
        self.df_valid = self.data[self.data.kfold == self.fold].reset_index(drop=True)

        self.x_train = self.df_train[self.features].values
        self.x_valid = self.df_valid[self.features].values
        if self.test is not None:
            self.x_test = self.test[self.features].values
        else:
            self.x_test = None

    def fit(self):
        self.model = ensemble.RandomForestClassifier(n_jobs=-1)

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)


class DecisionTreeModelSVD(DecisionTreeModel):
    def encode(self):
        super().encode()

        # FIX: We are not doing one hot encoding before svd!
        self.x_train, self.x_valid, self.x_test = reduce_dimensions_svd(
            self.x_train, self.x_valid, 120, x_test=self.x_test
        )


class XGBoost(DecisionTreeModel):
    def fit(self):
        self.model = xgb.XGBClassifier(
            n_jobs=-1, verbosity=0  # , max_depth=5, n_estimators=200
        )

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)


class LightGBM(DecisionTreeModel):
    def fit(self):
        # taken from https://www.kaggle.com/code/phongnguyen1/distance-to-key-locations#LightGBM
        params = {
            # # "n_estimators": 6058,
            # # "num_leaves": 107,
            # # "min_child_samples": 19,
            # # "learning_rate": 0.004899729720251191,
            # # # "log_max_bin": 10,
            # # "colsample_bytree": 0.5094776453903889,
            # # "reg_alpha": 0.007603254267129311,
            # # "reg_lambda": 0.008134379186044243,
        }

        self.model = LGBMClassifier(metric="auc", **params)

        # fit model on training data
        self.model.fit(
            self.x_train,
            self.df_train.loc[:, self.target].values,
            eval_set=[(self.x_valid, self.df_valid[self.target].values)],
            callbacks=[lgbm.early_stopping(100, verbose=False)],
            verbose=False,
        )


# # class CatBoost(DecisionTreeModel):
# #     def fit(self):
# #         # https://www.kaggle.com/code/alexandershumilin/playground-series-s3-e1-catboost-xgboost-lgbm
# #         params = {
# #             "n_estimators": 15000,
# #             "early_stopping_rounds": 1000,
# #             "random_seed": 0,
# #         }

# #         # # self.model = CatBoostRegressor(**params)
# #         self.model = CatBoostRegressor(
# #             iterations=100_000, loss_function="RMSE", random_seed=0
# #         )

# #         # fit model on training data
# #         self.model.fit(
# #             self.x_train,
# #             self.df_train.loc[:, self.target].values,
# #             eval_set=[(self.x_valid, self.df_valid[self.target].values)],
# #             early_stopping_rounds=params["early_stopping_rounds"],
# #             verbose=0,
# #         )

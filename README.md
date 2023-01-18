# kaggle-ps-s3e2

Kaggle Playground Series 3, Episode 2 competition

NOTES:

* Catboost worked best after tunning, even better than autogluon but close
* Position 95 of 770 teams, top 12.3%

Tasks to obtain the best model:

* [x] Basic eda
* [x] Cross-validation
* [x] First training
* [x] First submission
* [x] Merge original dataset -> Doesn't seem to improve performance
* [x] Implement LightGBM and Catboost -> Improvement
* [x] Set order in smoke status -> Doesn't seem to improve performance
* [x] Use AutoGluon framework -> Improvement
* [ ] Implement Lasso regression
* [ ] Use ten folds
* [ ] Try original dataset without adding label for original/synthetic
* [ ] Implement logistic regression
* [ ] Remove 'Residence_type', 'bmi'
* [ ] Scale numerical variables between 0 and 1

## Train, validation & submission

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=lgbm]  # [rf|svd|xgb|lgbm|cb]
```

Submission is stored in outputs folder (see `config.py` for complete path)

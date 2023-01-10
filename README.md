# kaggle-ps-s3e2

Kaggle Playground Series 3, Episode 2 competition

NOTES

* TODO

Tasks to obtain the best model:

* [ ] Basic eda
* [ ] Cross-validation
* [ ] First training
* [ ] First submission

## Train, validation & submission

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=lgbm]  # [lr|xgb|lgbm|cb]
```

Submission is stored in outputs folder (see `config.py` for complete path)

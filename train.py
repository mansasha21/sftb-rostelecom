import os
import random
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from preparator import *


def fix_seed(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


if __name__ == "__main__":
    seed_value = 7575
    fix_seed(seed_value)


    train_df = pd.read_csv('data/train_catugra_v2.csv', sep=',')
    val_df = pd.read_csv('data/test_catugra_v2.csv', sep=',', index_col=0)

    cat_cols = train_df.select_dtypes(include=['object']).columns.values
    train_df[cat_cols] = train_df[cat_cols].fillna('NaN')
    val_df[cat_cols] = val_df[cat_cols].fillna('NaN')

    models = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)

    for train_ids, test_ids in skf.split(train_df.drop('label', axis=1), train_df.label):
        x_tr = train_df.drop('label', axis=1).iloc[train_ids]
        x_ts = train_df.drop('label', axis=1).iloc[test_ids]
        y_tr = train_df.label[train_ids]
        y_ts = train_df.label[test_ids]

        ctb = CatBoostClassifier(iterations=10, verbose=200, task_type='GPU', random_seed=seed_value, eval_metric='AUC')
        ctb.fit(x_tr, y_tr, cat_features=cat_cols, eval_set=(x_ts, y_ts), use_best_model=True, early_stopping_rounds=150)
        models.append(ctb)

    for i, model in enumerate(models):
        try:
            os.mkdir('models')
        except:
            pass

        pickle.dump(model, open(f'./models/model{i}.pckl', 'wb'))

    # label_val_df = val_df[['label']]



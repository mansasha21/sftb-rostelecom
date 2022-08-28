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
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


if __name__ == "__main__":
    seed_value = 7575
    is_catboost = True
    fix_seed(seed_value)

    print("Train started")
    preparator = DataPreparator()
    train_df = pd.read_csv('data/train.csv', sep=';')

    print("Train preparation data started")
    preparator.fit(train_df, train_df.label, is_clustering=False)
    train_df = preparator.transform(df=train_df,
                                    add_region_statistical_data=True,
                                    add_rt_tariff_data=False,
                                    add_covid_data=False,
                                    fill_missing_categorical_by='NaN',
                                    fill_missing_numerical_by=np.min,
                                    type_data='train')
    print("Train preparation data ended")

    cat_cols = train_df.select_dtypes(include=['object']).columns.values

    models = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)

    print("Train models started")
    if is_catboost:
        for train_ids, test_ids in skf.split(train_df.drop('label', axis=1), train_df.label):
            x_tr = train_df.drop('label', axis=1).iloc[train_ids]
            x_ts = train_df.drop('label', axis=1).iloc[test_ids]
            y_tr = train_df.label[train_ids]
            y_ts = train_df.label[test_ids]
            ctb = CatBoostClassifier(iterations=10, verbose=200, task_type='GPU', random_seed=seed_value, eval_metric='AUC')
            ctb.fit(x_tr, y_tr, cat_features=cat_cols, eval_set=(x_ts, y_ts), use_best_model=True,
                    early_stopping_rounds=150)
            models.append(ctb)

        for i, model in enumerate(models):
            try:
                os.mkdir('models')
            except:
                pass

            pickle.dump(model, open(f'./models/model{i}.pckl', 'wb'))
    else:
        def roc_auc_my(y_true, y_pred):
            return 2 * roc_auc_score(y_true, y_pred) - 1

        # from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
        # from lightautoml.tasks import Task

        # automl = TabularUtilizedAutoML(
        #     timeout=3500,
        #     general_params={'nested_cv': False,
        #                     'use_algos': [['linear_l2', "lgb", "lgb_tuned", "cb", "cb_tuned"], ['lgb', 'linear_l2']]},
        #     reader_params={'cv': 10, 'random_state': seed_value, 'stratify': "label"},
        #     task=Task(
        #         name='binary',
        #         metric=roc_auc_my
        #     )
        # )
        # oof_pred = automl.fit_predict(
        #     train_df,
        #     roles={'target': 'label'}, verbose=2
        # )
    print("Train ended")

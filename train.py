import os
import random
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from preparator import *
import argparse


def fix_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


if __name__ == "__main__":
    # Seed everything
    seed_value = 7575
    fix_seed(seed_value)

    # Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        required=False,
                        default='catboost',
                        choices=['catboost', 'lama'],
                        help='type of used model')
    parser.add_argument('--train-path',
                        type=str,
                        required=False,
                        default='data/train.csv',
                        help='path to train data')
    parser.add_argument('--save-folder',
                        type=str,
                        required=False,
                        default='./models/',
                        help='path to save folder')
    parser.add_argument('--use-clustering',
                        type=bool,
                        required=False,
                        default=False,
                        help='use clustering feature')
    parser.add_argument('--add-region-statistical-data',
                        type=bool,
                        required=False,
                        default=True,
                        help='use region statistical feature')
    parser.add_argument('--add-rt-tariff-data',
                        type=bool,
                        required=False,
                        default=False,
                        help='use rt tariff feature')
    parser.add_argument('--add-covid-data',
                        type=bool,
                        required=False,
                        default=False,
                        help='use covid data feature')

    args = parser.parse_args()

    print("Train started")
    preparator = DataPreparator()
    train_df = pd.read_csv(args.train_path, sep=';')

    print("Train preparation data started")
    preparator.fit(train_df, train_df.label, is_clustering=args.use_clustering)
    train_df = preparator.transform(df=train_df,
                                    add_region_statistical_data=args.add_region_statistical_data,
                                    add_rt_tariff_data=args.add_rt_tariff_data,
                                    add_covid_data=args.add_covid_data,
                                    fill_missing_categorical_by='NaN',
                                    fill_missing_numerical_by=np.min,
                                    type_data='train')
    pickle.dump(preparator, open(str(args.save_folder) + 'preparator_' +
                                 str(preparator.add_region_statistical_data) + "_" + \
                                 str(preparator.add_rt_tariff_data) + "_" + \
                                 str(preparator.add_covid_data) + "_" + \
                                 str(preparator.fill_missing_categorical_by) + "_" + \
                                 str(preparator.is_cluster) + ".pkcle", 'wb'))
    print("Train preparation data ended")

    cat_cols = train_df.select_dtypes(include=['object']).columns.values

    models = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)

    print("Train models started")
    if args.model_name == "catboost":
        for train_ids, test_ids in skf.split(train_df.drop('label', axis=1), train_df.label):
            x_tr = train_df.drop('label', axis=1).iloc[train_ids]
            x_ts = train_df.drop('label', axis=1).iloc[test_ids]
            y_tr = train_df.label[train_ids]
            y_ts = train_df.label[test_ids]
            ctb = CatBoostClassifier(iterations=10,
                                     verbose=200,
                                     task_type='GPU',
                                     random_seed=seed_value,
                                     use_best_model=True,
                                     eval_metric='NormalizedGini',  # AUC
                                     loss_function='Logloss',
                                     early_stopping_rounds=500,
                                     )
            ctb.fit(x_tr,
                    y_tr,
                    cat_features=cat_cols,
                    eval_set=(x_ts, y_ts))
            models.append(ctb)

        for i, model in enumerate(models):
            try:
                os.mkdir('models')
            except:
                pass
            pickle.dump(model, open(f'{args.save_folder}/model{i}.pckl', 'wb'))
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

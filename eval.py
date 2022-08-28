import argparse
import os
import pickle
from preparator import *
import pandas as pd

pd.options.mode.chained_assignment = None
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        required=False,
                        default='catboost',
                        choices=['catboost', 'lama'],
                        help='type of used model')
    parser.add_argument('--val-path',
                        type=str,
                        required=False,
                        default='data/test.csv',
                        help='path to test data')
    parser.add_argument('--models-dir',
                        type=str,
                        required=False,
                        default='./models/',
                        help='path to folder with models')
    parser.add_argument('--result-path',
                        type=str,
                        required=False,
                        default='./results/',
                        help='path to processed result')
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
    parser.add_argument('--add-rozn-data',
                        type=bool,
                        required=False,
                        default=True,
                        help='use rozn data feature')
    args = parser.parse_args()

    print("Evaluation started")
    preparator = automl = pickle.load(open(str(args.models_dir) + 'preparator_' +
                                           str(args.add_region_statistical_data) + "_" + \
                                           str(args.add_rt_tariff_data) + "_" + \
                                           str(args.add_covid_data) + "_" + \
                                           "NaN" + "_" + \
                                           str(args.use_clustering) + ".pkcle", 'rb'))
    print("Start data preparation")

    val_df = pd.read_csv(args.val_path, sep=';')
    val_df = preparator.transform(df=val_df,
                                  add_region_statistical_data=args.add_region_statistical_data,
                                  add_rt_tariff_data=args.add_rt_tariff_data,
                                  add_covid_data=args.add_covid_data,
                                  add_rozn_data=args.add_rozn_data,
                                  fill_missing_categorical_by='NaN',
                                  fill_missing_numerical_by=np.min,
                                  type_data='test')
    print("End data preparation")

    cat_cols = val_df.select_dtypes(include=['object']).columns.values
    preds = None
    if args.model_name == "catboost":
        preds = np.mean(
            [pickle.load(open(model_path, 'rb')).predict_proba(val_df.drop('label', axis=1))[:, 1] for model_path in
             Path(str(args.models_dir)).glob('*.pckl')], axis=0)
    else:
        # preds = np.mean(
        #     [pickle.load(open(model_path, 'rb')).predict_proba(val_df.drop('label', axis=1))[:, 1] for model_path in
        #      models_dir.glob('*.pckl')], axis=0)
        #
        # automl =
        pass
    label_val_df = val_df[['label']]
    label_val_df['pred'] = preds
    top5p = int(label_val_df.shape[0] * 0.05)
    res_response = label_val_df.sort_values('pred', ascending=False).iloc[:top5p].label.sum() / top5p
    print('\n\n\n\n')
    print(res_response)
    try:
        os.mkdir(args.result_path)
    except:
        pass
    label_val_df.sort_values('pred', ascending=False).iloc[:top5p]["pred"].to_csv(args.result_path + "preds.csv")

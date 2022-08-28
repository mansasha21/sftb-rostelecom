import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from pathlib import Path
from preparator import *

if __name__ == "__main__":
    models_dir = Path('./models')

    # TODO сделать нормальный подсос данных
    train_df = pd.read_csv('data/train_catugra_v2.csv', sep=',')
    val_df = pd.read_csv('data/test_catugra_v2.csv', sep=',', index_col=0)

    cat_cols = train_df.select_dtypes(include=['object']).columns.values
    train_df[cat_cols] = train_df[cat_cols].fillna('NaN')
    val_df[cat_cols] = val_df[cat_cols].fillna('NaN')

    # train_df = process_period(train_df)
    # val_df = process_period(val_df)

    preds = np.mean([pickle.load(open(model_path, 'rb')).predict_proba(val_df.drop('label', axis=1))[:,1] for model_path in models_dir.glob('*.pckl')], axis=0)

    label_val_df = val_df[['label']]
    label_val_df['pred'] = preds
    top5p = int(label_val_df.shape[0] * 0.05)
    res_response = label_val_df.sort_values('pred', ascending=False).iloc[:top5p].label.sum()/top5p
    print('\n\n\n\n')
    print(res_response)
    label_val_df.sort_values('pred', ascending=False).iloc[:top5p]["pred"].to_csv('ans.csv')
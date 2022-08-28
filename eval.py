import pickle
from pathlib import Path
from preparator import *
import pandas as pd
pd.options.mode.chained_assignment = None
if __name__ == "__main__":
    models_dir = Path('./models')
    is_catboost = True

    print("Evaluation started")
    preparator = automl = pickle.load(open(models_dir / 'preparator_True_False_False_NaN_False.pkcle', 'rb'))
    print("Start data preparation")

    val_df = pd.read_csv('data/test.csv', sep=';')
    val_df = preparator.transform(df=val_df,
                                  add_region_statistical_data=True,
                                  add_rt_tariff_data=False,
                                  add_covid_data=False,
                                  fill_missing_categorical_by='NaN',
                                  fill_missing_numerical_by=np.min,
                                  type_data='test')
    print("End data preparation")

    cat_cols = val_df.select_dtypes(include=['object']).columns.values
    preds = None
    if is_catboost:
        preds = np.mean([pickle.load(open(model_path, 'rb')).predict_proba(val_df.drop('label', axis=1))[:,1] for model_path in models_dir.glob('*.pckl')], axis=0)
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
    res_response = label_val_df.sort_values('pred', ascending=False).iloc[:top5p].label.sum()/top5p
    print('\n\n\n\n')
    print(res_response)
    label_val_df.sort_values('pred', ascending=False).iloc[:top5p]["pred"].to_csv('ans.csv')
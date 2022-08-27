from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder
from category_encoders import TargetEncoder
from category_encoders import OrdinalEncoder


class DataPreparator:
    def __init__(self):
        # self.data = data
        self.is_cluster = False
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.ordinal_encoder = OrdinalEncoder()
        self.label_encoder = LabelEncoder()
        self.cat_boost_encoder = CatBoostEncoder()
        self.target_encoder = TargetEncoder()
        self.clustelizer = None

    def transform(self, df,
                  fill_missing_categorical_by="mode",
                  fill_missing_numerical_by=np.mean,
                  type_data='train'):
        """
        Transform the data to the model
        :param df: Dataframe to transform
        :param fill_missing_categorical_by: Should fill missing categorical values
        :param fill_missing_numerical_by: Should fill missing numerical values
        """
        new_df = df.copy()
        # Fill categorical missing values
        cat_cols = new_df.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            if fill_missing_categorical_by == "NaN":
                new_df[col] = new_df[col].fillna('NaN')
            else:
                new_df[col] = new_df.groupby('subject_name')[col].transform(lambda x: x.fillna(x.mode()))

        # Fill numerical missing values
        num_cols = new_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # new_df[num_cols] = new_df[num_cols].fillna("NaN")
        for col in num_cols:
            new_df[col] = new_df[col].fillna(new_df.groupby('subject_name')[col].transform(np.mean))
        for col in num_cols:
            new_df[col] = new_df[col].fillna(new_df.groupby('district')[col].transform(np.mean))

        # TODO: как то переписать type_data
        # TODO: add month and powertranform
        f_cols = [i for i in num_cols if i.startswith('f')]
        if self.is_cluster and type_data == 'train':
            self._fit_cluster(new_df[f_cols])
            new_df['cluster'] = self._clusterize_data_(new_df[f_cols])
        elif self.is_cluster and type_data == 'test':
            new_df['cluster'] = self._clusterize_data_(new_df[f_cols])
        return new_df

    def fit(self,
            X,
            y,
            is_clustering=True,
            type_of_scaler=None,
            type_of_encoder=None,
            ):
        """
        Fit the data to the model
        :param X: Dataframe to fit
        :param y: Target to fit
        :param is_clustering: Is the model a clustering model
        :param type_of_scaler: Type of scaler to use
        :param type_of_encoder: Type of encoder to use
        """
        if type_of_scaler == "standard":
            self.standard_scaler.fit(X)
        elif type_of_scaler == "min_max":
            self.min_max_scaler.fit(X)

        encoder = {"ordinal": self.ordinal_encoder,
                   "label": self.label_encoder,
                   "cat_boost": self.cat_boost_encoder,
                   "target": self.target_encoder
                   }
        encoder[type_of_encoder].fit(X, y)

        if is_clustering:
            self.is_cluster = True

    def _fit_cluster(self, df: pd.DataFrame):
        BGM = BayesianGaussianMixture(n_components=7, covariance_type='full', max_iter=300, n_init=5)
        BGM.fit(df)
        self.clustelizer = BGM

    def _clusterize_data_(self, df: pd.DataFrame):
        cluster_predict = self.clustelizer.predict(df)
        return cluster_predict

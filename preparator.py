from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder
from category_encoders import TargetEncoder
from category_encoders import OrdinalEncoder


class DataPreparator:
    def __init__(self, data):
        self.data = data
        self.is_cluster = False
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.ordinal_encoder = OrdinalEncoder()
        self.label_encoder = LabelEncoder()
        self.cat_boost_encoder = CatBoostEncoder()
        self.target_encoder = TargetEncoder()

    def transform(self, df,
                  fill_missing_categorical_by="mode",
                  fill_missing_numerical_by="mode"):
        """
        Transform the data to the model
        :param df: Dataframe to transform
        :param fill_missing_categorical_by: Should fill missing categorical values
        :param fill_missing_numerical_by: Should fill missing numerical values
        """
        # Fill categorical missing values
        cat_cols = df.select_dtypes(include=['object']).columns.values
        for col in cat_cols:
            if fill_missing_categorical_by == "NaN":
                df[col] = df[col].fillna('NaN')
            else:
                df[col] = df.groupby('subject_name')[col].transform(lambda x: x.fillna(x.mode()))

        # Fill numerical missing values
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.values
        for col in num_cols:
            df[col] = df.groupby('subject_name')[col].transform(fill_missing_numerical_by)
        if self.is_cluster:
            df['cluster'] = self._clusterize_data_(df)
        pass

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

        # TODO: add clustering
        if is_clustering:
            pass

    def _clusterize_data_(self, df):
        pass

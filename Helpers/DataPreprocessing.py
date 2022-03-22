from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, top, dummy_encode=True):
        self.top = top
        self.dummy_encode = dummy_encode
        self.all_cat_columns_in_raw_data_hardcoded = ['Airline',
                                                      'Destination',
                                                      'AircraftType',  # if actually used remember to change all rows to string outside
                                                      'FlightType',
                                                      'Sector'
                                                      ]

    def scale(self, X, columns):
        X_copy = X.copy()
        val = X[columns].values

        if len(columns) == 1:
            val = val.reshape(-1, 1)

        scaler = StandardScaler().fit(val)
        features = scaler.transform(val)

        X_copy[columns] = features
        return X_copy

    def fit(self, X, y=None):
        self.df = X.copy()
        self.attributes = np.asarray(self.df.columns)
        # Find categorical variables
        self.categorical_attributes_used = []
        self.__populate_cat_attribs_array()
        # Find top values for each categorical attribute
        self.cat_top_dict = {}
        self.__populate_cat_top_dict()

    def transform(self, X, columns):
        df = X.copy()
        df = df.sort_values(by=['ScheduleTime'])
        df = df.reset_index(drop=True)
        df = self.__fix_flight_type(df)
        df = self.__prepare_datetime_data(df)
        df = self.__categorical_data_transform(df)
        # add columns (For instance add SeatCapacity)
        df = self.__add_Columns_To_df(X, df, columns)
        return df

    def __populate_cat_attribs_array(self):
        for attr in self.attributes:
            if attr in self.all_cat_columns_in_raw_data_hardcoded:
                if attr not in self.categorical_attributes_used:
                    self.categorical_attributes_used.append(attr)

    def __populate_cat_top_dict(self):
        for attr in self.categorical_attributes_used:
            if attr in self.top:
                top = self.top[attr]
                attr_len = len(self.df[attr].unique())
                if attr_len < top:
                    top = attr_len
                self.cat_top_dict[attr] = np.asarray(
                    self.df[attr].value_counts()[:top].index)

    def __fix_flight_type(self, df):
        df = df.copy()
        attributes = np.asarray(df.columns)
        if ('FlightType' in attributes) and ('FlightType' in self.categorical_attributes_used):
            # G is a passenger flight. Lectures
            df['FlightType'].replace('G', 'J', inplace=True)
            # O is a charter type. Lectures
            df['FlightType'].replace('O', 'C', inplace=True)
            return df

    def __prepare_datetime_data(self, df):
        df = df.copy()
        if 'ScheduleTime' in self.attributes:
            df['ScheduleTime'] = pd.to_datetime(df['ScheduleTime'])
            df['Year'] = df['ScheduleTime'].map(lambda x: x.year)
            df['Month'] = df['ScheduleTime'].map(lambda x: x.month)
            df['WeekNumber'] = df['ScheduleTime'].map(lambda x: int(x.strftime("%W")))
            df['Day'] = df['ScheduleTime'].map(lambda x: x.day)
            df['Hour'] = df['ScheduleTime'].map(lambda x: x.hour)
            df['Weekday'] = df['ScheduleTime'].map(lambda x: x.weekday())
            df['Holiday'] = self.__get_holidays(df)
            df.drop('ScheduleTime', axis=1, inplace=True)
            return df

    # 1) Gregorian calendar
    # 2) Christmas Holiday
    def __get_holidays(self, df, easter_days_off=3, christmas_weeks_numbers=[51, 52, 53]):
        # Easter
        import dateutil.easter
        years = np.asarray(df.ScheduleTime.dt.year)
        easter_index = [dateutil.easter.easter(year) for year in years]
        easter_index = np.array(np.abs(
            easter_index - df.ScheduleTime.dt.date).dt.days <= easter_days_off)

        # Christmas - New Year's Eve
        weeks = np.asarray(df.ScheduleTime.dt.isocalendar().week)
        christmas_index = np.array(
            [weeks[i] in christmas_weeks_numbers for i in range(df.shape[0])])

        # merge the indixes
        indx = christmas_index | easter_index
        return indx.astype(int)

    def __categorical_data_transform(self, df):
        df = df.copy()
        for attr in self.categorical_attributes_used:
            temp_df = self.__get_top_values_df(df, attr)
            df.drop([attr], axis=1, inplace=True)
            df = pd.concat([df, temp_df], axis=1)
            if self.dummy_encode:
                df = pd.get_dummies(df, columns=[attr],
                                    prefix=attr, drop_first=False)
                df = self.__remove_others(df)
        return df

    def __get_top_values_df(self, df, column, replacement_name='Other'):
        df = df[column].copy()
        top_values = self.cat_top_dict[column]
        top = self.top[column]
        if len(top_values) < top:
            top = len(top_values)
        df[df.isin(top_values[:top]) == False] = replacement_name
        return pd.DataFrame(data=df, columns=[column])

    def __remove_others(self, df):
        df = df.copy()
        column_names = np.asarray(df.columns)
        for col in column_names:
            if len(col) > 6:
                end_name = col[-5:].lower()
                if end_name == "other":
                    df.drop([col], axis=1, inplace=True)
        return df

    def __add_Columns_To_df(self, X, df, columns):
        """
        Concatenate the columns of X to df.

        ====
        X: Dataframe that contains the columns
        df: The Dataframe in which we will concatenate the specified columns
        columns: A list of columns from X to be concatenated to df

        ===
        returns
        df: the initial df after concatenation
        """

        for column in columns:
            df = pd.concat([df, X[column]], axis=1)
        return df

    def __target_encode_column_df(self, X, target, column):
        X_copy = X.copy()
        enc_map = dict()
        total_target_mean = np.mean(X[target])
        column_mean = X.groupby([column])[target].mean()
        X[column] = X[column].map(column_mean)
        enc_map = dict(zip(X_copy[column], X[column]))
        return X, enc_map

    def __target_encode_column_dict_df(self, X, enc_map, column):
        X[column] = X[column].apply(lambda x: str(enc_map.get(x)))
        return X
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class AtemporalEncodingFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df, attributes_used, categorical_attributes_used,  top_value_counts, easter_days_off=3, christmas_weeks_numbers=[52, 53, 1]):
        self.df = df[attributes_used].copy(deep=True)
        self.attributes_used = attributes_used.copy()
        if 'FlightType' in self.attributes_used:
            self.__fix_flight_type()
        self.categorical_attributes_used = categorical_attributes_used.copy()
        self.top_value_counts = top_value_counts
        self.easter_days_off = easter_days_off
        self.christmas_weeks_numbers = christmas_weeks_numbers
        # self.cat_dict = {}
        self.cat_dict = self.__get_categorical_encoding()

    def fit(self, X, y=None):
        # self.cat_dict = self.__get_categorical_encoding()
        return self

    def transform(self, X):
        if 'FlightType' in self.attributes_used:
            self.__fix_flight_type()
        if 'FlightType' in self.cat_dict.keys():
            self.cat_dict['FlightType'] = self.cat_dict['FlightType'][0]
        X = self.__prepare_data(self.cat_dict)
        if 'LoadFactor' not in self.df.columns:
            return X
        y = pd.DataFrame(self.df['LoadFactor'])
        data = pd.concat((X, y), axis=1)
        return data

    def __fix_flight_type(self):
        # G is a passenger flight. Lectures
        self.df['FlightType'].replace('G', 'J', inplace=True)
        # O is a charter type. Lectures
        self.df['FlightType'].replace('O', 'C', inplace=True)

    def __get_categorical_encoding(self):
        cat_dict = {}
        for attr in self.categorical_attributes_used:
            top = self.top_value_counts
            if len(self.df[attr].value_counts()) < top:
                top = len(self.df[attr].value_counts())
            cat_dict[attr] = self.df[attr].value_counts().head(top).index
        return cat_dict

    def __get_holidays(self):
        # Easter
        import dateutil.easter

        years = np.asarray(self.df.ScheduleTime.dt.year)
        easter_index = [dateutil.easter.easter(year) for year in years]
        easter_index = np.array(np.abs(
            easter_index - self.df.ScheduleTime.dt.date).dt.days <= self.easter_days_off)

        # Christmas - New Year's Eve
        weeks = np.asarray(self.df.ScheduleTime.dt.isocalendar().week)
        christmas_index = np.array(
            [weeks[i] in self.christmas_weeks_numbers for i in range(self.df.shape[0])])

        # merge the indixes
        indx = christmas_index | easter_index
        temp_df = {'Holiday': indx.astype(int)}
        temp_df = pd.DataFrame(temp_df)
        return indx.astype(int)

    def __prepare_data(self, cat_dict):
        if 'ScheduleTime' not in self.attributes_used:
            print('ScheduleTime not in attributes list')
            return
        self.df['Year'] = self.df['ScheduleTime'].dt.isocalendar().year
        self.df['WeekNumber'] = self.df['ScheduleTime'].dt.isocalendar().week
        self.df['Day'] = self.df['ScheduleTime'].dt.isocalendar().day
        self.df['Hour'] = self.df['ScheduleTime'].dt.hour
        self.df['Holiday'] = self.__get_holidays()
        self.df.drop('ScheduleTime', axis=1, inplace=True)

        if cat_dict is None:
            return self.df

        for attr in cat_dict.keys():
            for i in cat_dict[attr]:
                feature_name = attr + '_' + str(i)
                self.df[feature_name] = (self.df[attr] == i).astype(int)
            self.df.drop(attr, axis=1, inplace=True)
        return self.df  # Not the most useful


class AtemporalDataTreeTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, df, attributes_used, categorical_attributes_used, top=10):
        self.df = df[attributes_used].copy(deep=True)
        self.attributes_used = attributes_used
        self.top = top
        self.categorical_attributes_used = categorical_attributes_used
        self.df = self.__fix_flight_type(self.df)
        self.top_cat_dict = {}
        self.__create_top_dict()

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df[self.attributes_used].copy()
        df = self.__fix_flight_type(df)
        df = self.__prepare_datetime_data(df)
        df = self.__categorical_data_transform(df)
        return df

    def __fix_flight_type(self, df):
        df = df.copy()
        # G is a passenger flight. Lectures
        df['FlightType'].replace('G', 'J', inplace=True)
        # O is a charter type. Lectures
        df['FlightType'].replace('O', 'C', inplace=True)
        return df

    def __get_top_values_df(self, df, column, dictionary_of_cat_values, replacement_name='Other'):
        df = df[column].copy(deep=True)
        top_values = dictionary_of_cat_values[column]
        top = self.top
        if len(top_values) < top:
            top = len(top_values)
        df[df.isin(top_values[:top]) == False] = replacement_name
        return pd.DataFrame(data=df, columns=[column])

    def __create_top_dict(self):
        for attr in self.categorical_attributes_used:
            self.top_cat_dict[attr] = np.asarray(
                self.df[attr].value_counts().index)

    def __categorical_data_transform(self, df):
        df = df.copy()
        for attr in self.categorical_attributes_used:
            temp_df = self.__get_top_values_df(df, attr, self.top_cat_dict)
            df.drop([attr], axis=1, inplace=True)
            df = pd.concat([df, temp_df], axis=1)
        return df

    def __prepare_datetime_data(self, df):
        df = df.copy()
        if 'ScheduleTime' not in self.attributes_used:
            print('ScheduleTime not in attributes list')
            return
        df['Year'] = df['ScheduleTime'].dt.isocalendar().year
        df['WeekNumber'] = df['ScheduleTime'].dt.isocalendar().week
        df['Day'] = df['ScheduleTime'].dt.isocalendar().day
        df['Hour'] = df['ScheduleTime'].dt.hour
        df['Holiday'] = self.__get_holidays()
        df.drop('ScheduleTime', axis=1, inplace=True)

        return df

    def __get_holidays(self, easter_days_off=3, christmas_weeks_numbers=[51, 52, 53]):
        # Easter
        import dateutil.easter

        years = np.asarray(self.df.ScheduleTime.dt.year)
        easter_index = [dateutil.easter.easter(year) for year in years]
        easter_index = np.array(np.abs(
            easter_index - self.df.ScheduleTime.dt.date).dt.days <= easter_days_off)

        # Christmas - New Year's Eve
        weeks = np.asarray(self.df.ScheduleTime.dt.isocalendar().week)
        christmas_index = np.array(
            [weeks[i] in christmas_weeks_numbers for i in range(self.df.shape[0])])

        # merge the indixes
        indx = christmas_index | easter_index
        temp_df = {'Holiday': indx.astype(int)}
        temp_df = pd.DataFrame(temp_df)
        return indx.astype(int)

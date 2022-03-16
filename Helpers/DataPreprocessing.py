from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class AtemporalEncodingFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df, attributes_used, categorical_attributes_used,  top_value_counts, easter_days_off=3, christmas_weeks_numbers=[52, 53, 1]):
        self.df = df[attributes_used].copy(deep=True)
        self.attributes_used = attributes_used.copy()
        self.categorical_attributes_used = categorical_attributes_used.copy()
        self.top_value_counts = top_value_counts
        self.easter_days_off = easter_days_off
        self.christmas_weeks_numbers = christmas_weeks_numbers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cat_dict = self.__get_categorical_encoding()
        if 'FlightType' in cat_dict.keys():
            cat_dict['FlightType'] = cat_dict['FlightType'][:-2]
        X = self.__prepare_data(cat_dict)
        if 'LoadFactor' not in self.df.columns:
            return X
        y = pd.DataFrame(self.df['LoadFactor'])
        data = pd.concat((X, y), axis=1)
        return data

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

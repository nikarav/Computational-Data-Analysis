import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataTransformer():
    def __init__(self, top_percent, dummy_encode=True):
        self.top_percent = top_percent
        self.dummy_encode = dummy_encode

    def fit(self, X):
        self.df = X.copy()
        self.attributes = np.asarray(self.df.columns)
        # Find categorical variables
        self.categorical_attributes_used = [col for col in np.asarray(
            self.df.columns) if self.df[col].dtype == 'O']
        # Find top values for each categorical attribute
        self.top_percent = self.__get_top_list(
            self.top_percent, self.categorical_attributes_used)
        self.df = self.__fix_flight_type(self.df)
        self.cat_top_dict = {}
        # self.__populate_cat_top_dict()
        self.__populate_cat_top_dict_selective()

    def transform(self, X):
        df = X.copy()
        #df = df.sort_values(by=['ScheduleTime'])
        df = self.__fix_flight_type(df)
        df = self.__prepare_datetime_data(df)
        df = self.__categorical_data_transform(df)
        return df

    def select_top_features(self, df, column, percent):
        value_counts = df[column].value_counts(normalize=True).cumsum().values
        variables = df[column].value_counts(normalize=True).cumsum().index
        array_to_ret = np.asarray(
            variables[np.where(value_counts <= percent/100)])
        if len(array_to_ret) == 0:
            return np.array([np.asarray(variables[np.where(value_counts >= percent/100)])[0]])
        return array_to_ret

    def __populate_cat_top_dict(self):
        for i, attr in enumerate(self.categorical_attributes_used):
            top = self.top_percent[i]
            top_attribute_index = int(
                top/100*len(set(self.df[attr].value_counts().values)))
            if top_attribute_index == 0:
                top_attribute_index = 1
            self.cat_top_dict[attr] = np.asarray(self.df[attr].value_counts()[
                :top_attribute_index].index)

    def __populate_cat_top_dict_selective(self, columns=['Airline', 'AircraftType', 'FlightType'], top_percent=[85, 85, 80]):
        for i, attr in enumerate(columns):
            top = top_percent[i]
            top_attributes = self.select_top_features(self.df, attr, top)
            self.cat_top_dict[attr] = top_attributes

    def __fix_flight_type(self, df):
        df = df.copy()
        attributes = np.asarray(df.columns)
        if ('FlightType' in attributes) and ('FlightType' in self.categorical_attributes_used):
            # G is a passenger flight. Lectures
            df['FlightType'].replace('G', 'J', inplace=True)
            # O is a charter type. Lectures
            df['FlightType'].replace('O', 'C', inplace=True)
            return df

    # def __prepare_datetime_data2(self, df):
    #     df = df.copy()
    #     if 'ScheduleTime' in self.attributes:
    #         df['Year'] = df['ScheduleTime'].dt.year
    #         df['WeekNumber'] = df['ScheduleTime'].dt.isocalendar().week
    #         df['Day'] = df['ScheduleTime'].dt.isocalendar().day
    #         df['Month'] = df['ScheduleTime'].dt.month
    #         df['Hour'] = df['ScheduleTime'].dt.hour
    #         df['Days_from_Easter'] = self.__get_days_from_easter(df)
    #         df['Days_from_new_year'] = self.__get_days_from_new_years(df)
    #         df.drop('ScheduleTime', axis=1, inplace=True)
    #         return df

    def __prepare_datetime_data(self, df):
        df = df.copy()
        if 'ScheduleTime' in self.attributes:
            df['ScheduleTime'] = pd.to_datetime(df['ScheduleTime'])
            df['Year'] = df['ScheduleTime'].map(lambda x: x.year)
            df['Month'] = df['ScheduleTime'].map(lambda x: x.month)
            df['WeekNumber'] = df['ScheduleTime'].map(
                lambda x: int(x.strftime("%W")))
            df['Day'] = df['ScheduleTime'].map(lambda x: x.day)
            df['Hour'] = df['ScheduleTime'].map(lambda x: x.hour)
            df['Weekday'] = df['ScheduleTime'].map(lambda x: x.weekday())
            df['Days_from_Easter'] = self.__get_days_from_easter(df)
            df['Days_from_new_year'] = self.__get_days_from_new_years(df)
            df.drop('ScheduleTime', axis=1, inplace=True)
            return df

    def __get_days_from_easter(self, df):
        import dateutil.easter
        years = np.asarray(df.ScheduleTime.dt.year)
        easter_dates = [dateutil.easter.easter(year) for year in years]
        days_off = np.array(
            np.abs(easter_dates - df.ScheduleTime.dt.date).dt.days)
        return days_off

    def __get_days_from_new_years(self, df):
        import datetime
        years = np.asarray(df.ScheduleTime.dt.year)
        new_years_days = [datetime.date(year, 1, 1) for year in years]
        new_years_days_p1 = [datetime.date(year+1, 1, 1) for year in years]
        days_off = np.abs(new_years_days - df.ScheduleTime.dt.date).dt.days
        days_off_p1 = np.abs(new_years_days_p1 -
                             df.ScheduleTime.dt.date).dt.days
        days = pd.concat([days_off, days_off_p1], axis=1)
        return days.min(axis=1)  # otherwise just days_off_p1

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

    def __categorical_data_transform(self, df, columns=['Airline', 'AircraftType', 'FlightType']):
        df = df.copy()
        for attr in columns:
            temp_df = self.__get_top_values_df(df, attr)
            df.drop([attr], axis=1, inplace=True)
            df = pd.concat([df, temp_df], axis=1)
            if self.dummy_encode:
                df = pd.get_dummies(
                    df, columns=[attr], prefix=attr, drop_first=False)
                df = self.__remove_others(df)
        return df

    def __categorical_data_transform2(self, df, columns):
        df = df.copy()
        for attr in columns:
            temp_df = self.__get_top_values_df(df, attr)
            df.drop([attr], axis=1, inplace=True)
            df = pd.concat([df, temp_df], axis=1)
            if self.dummy_encode:
                df = pd.get_dummies(
                    df, columns=[attr], prefix=attr, drop_first=False)
                df = self.__remove_others(df)
        return df

    def get_top_variables(self, df, columns, top_percent):
        df = df.copy()
        cat_dict = {}
        for i, cat in enumerate(columns):
            cat_dict[cat] = self.select_top_features(df, cat, top_percent[i])

        return df, cat_dict

    def __get_top_values_df(self, df, column, replacement_name='Other'):
        df = df[column].copy()
        top_values = self.cat_top_dict[column]
        df[df.isin(top_values) == False] = replacement_name
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

    def __get_top_list(self, top_percent, cat_features):
        if type(top_percent) is int or type(top_percent) is float:
            top = [top_percent]*len(cat_features)
            return top
        elif type(top_percent) is list:
            len_top_pc = len(top_percent)
            len_cat_feat = len(cat_features)
            if len_top_pc == len_cat_feat:
                top = top_percent
                return top
            if len_top_pc < len_cat_feat:
                top = top_percent + [top_percent[-1]] * \
                    (len_cat_feat - len_top_pc)
                return top
            if len_top_pc > len_cat_feat:
                top = [top_percent[i] for i in range(len_cat_feat)]
                return top
        else:
            print('Top percent should be float, int or array type')
            return None

    def target_encode_column_df(self, X, target, column):
        X_copy = X.copy()
        enc_map = dict()
        total_target_mean = np.mean(X[target])
        column_mean = X.groupby([column])[target].mean()
        X[column] = X[column].map(column_mean)
        enc_map = dict(zip(X_copy[column], X[column]))
        return X, enc_map

    def target_encode_column_dict_df(self, X, enc_map, column):
        X[column] = X[column].apply(lambda x: enc_map.get(x))
        return X

    def scale(self, X, columns):
        X_copy = X.copy()
        val = X[columns].values

        val = val.reshape(-1, 1)

        scaler = StandardScaler().fit(val)
        features = scaler.transform(val)

        X_copy[columns] = features
        return X_copy


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class SamplingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sample_by='weeks', horizon=4, memory=8, target_passengers_number=True):
        self.sample_by = sample_by.lower()
        if sample_by not in ['hours', 'days', 'weeks']:
            self.sample_by = 'days'
        self.horizon = np.abs(horizon)
        self.memory = np.abs(memory)
        self.target_passengers_number = target_passengers_number

    def fit(self, X):
        X = X.copy(deep=True)
        df, indexes = self.transform(X)
        if 'NumberOfPassengers' not in np.asarray(df.columns):
            return df, indexes
        X = df.drop(["NumberOfPassengers"], axis=1)
        y = df[['NumberOfPassengers']]
        return X, indexes, y

    def transform(self, X):
        df = X.copy(deep=True)
        if 'ScheduleTime' not in np.asarray(df.columns):
            print('Input data lacks Schedule Time columns!')
            return
        df = df.sort_values(by=['ScheduleTime'])
        df = self.__fix_flight_type(df)
        df = self.__prepare_date_time_data(df)
        if self.sample_by == 'weeks':
            df, indexes = self.__group_by_week(df)
        if self.sample_by == 'days':
            df, indexes = self.__group_by_day(df)
        if self.sample_by == 'hours':
            df, indexes = self.__group_by_hour(df)
        return df, indexes

    def __fix_flight_type(self, df):
        df = df.copy()
        attributes = np.asarray(df.columns)
        if ('FlightType' in attributes):
            # G is a passenger flight. Lectures
            df['FlightType'].replace('G', 'J', inplace=True)
            # O is a charter type. Lectures
            df['FlightType'].replace('O', 'C', inplace=True)
            # df['FlightType'] = df['FlightType'] ## WTF did I write?
            return df

    def __prepare_date_time_data(self, df):
        df = df.copy(deep=True)
        df['Year'] = df['ScheduleTime'].dt.isocalendar().year
        df['Month'] = df['ScheduleTime'].dt.month
        df['WeekNumber'] = df['ScheduleTime'].dt.isocalendar().week
        df['Day'] = df['ScheduleTime'].dt.isocalendar().day
        df['Hour'] = df['ScheduleTime'].dt.hour
        df['Holiday'] = self.__get_holidays(df)
        df.drop('ScheduleTime', axis=1, inplace=True)
        if self.sample_by == 'days':
            df.drop('Hour', axis=1, inplace=True)
        if self.sample_by == 'weeks':
            df.drop('Hour', axis=1, inplace=True)
            df.drop('Day', axis=1, inplace=True)
        if 'LoadFactor' in np.asarray(df.columns):
            df['NumberOfPassengers'] = df['SeatCapacity'].values * \
                df['LoadFactor'].values
            df.drop('LoadFactor', axis=1, inplace=True)
            df.drop('SeatCapacity', axis=1, inplace=True)
        return df

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

    def __sample(self, df, y):
        df = df.copy()
        if self.sample_by == 'weeks':
            weekly_mean_passengers = []

    def __group_by_week(self, df):
        df = df.copy()
        flag = False
        indexes = []
        week_numbers = []
        month_number = []
        holiday_period = []
        number_of_flights = []
        number_of_destinations = []
        number_of_chartered = []
        number_of_sectors = []
        indices = []
        if 'NumberOfPassengers' in np.asarray(df.columns):
            flag = True
            mean_passenger_number = []
        years = np.array(df['Year'].unique())
        i = 0
        for year in years:
            year_df = df[df.Year == year]
            weeks = np.array(year_df.WeekNumber.unique())
            for week in weeks:
                week_numbers.append(week)
                week_df = year_df[year_df.WeekNumber == week]
                indexes.append(np.asarray(week_df.index))
                month_number.append(week_df.Month.value_counts().index[0])
                holiday_period.append(
                    week_df.Holiday.value_counts().index[0])
                number_of_flights.append(
                    len(np.asarray(week_df.FlightNumber.unique())))
                number_of_destinations.append(
                    len(np.asarray(week_df.Destination.unique())))
                number_of_chartered.append(
                    len(np.asarray(week_df[week_df.FlightType == 'C'])))
                number_of_sectors.append(
                    len(np.asarray(week_df.Sector.unique())))
                if flag:
                    mean_no_pass = np.mean(
                        week_df['NumberOfPassengers'].values)
                    mean_passenger_number.append(mean_no_pass)
                indices.append(i)
                i += 1
        week_numbers = np.array([week_numbers]).T
        month_number = np.array([month_number]).T
        holiday_period = np.array([holiday_period]).T
        number_of_flights = np.array([number_of_flights]).T
        number_of_destinations = np.array([number_of_destinations]).T
        number_of_chartered = np.array([number_of_chartered]).T
        number_of_sectors = np.array([number_of_sectors]).T
        indices = np.array([indices]).T
        if flag:
            mean_passenger_number = np.array([mean_passenger_number]).T
            data = np.concatenate([week_numbers, month_number, holiday_period, number_of_flights,
                                   number_of_destinations, number_of_chartered, number_of_sectors, mean_passenger_number], axis=1)
            df = pd.DataFrame(data=data, columns=[
                'WeekNumber', 'Month', 'Holiday', 'Flights', 'Destinations', 'Chartered', 'Sectors', 'NumberOfPassengers'])
            return df, indexes
        data = np.concatenate([week_numbers, month_number, holiday_period, number_of_flights,
                               number_of_destinations, number_of_chartered, number_of_sectors], axis=1)
        df = pd.DataFrame(data=data, columns=[
            'WeekNumber', 'Month', 'Holiday', 'Flights', 'Destinations', 'Chartered', 'Sectors'])
        return df, indexes

    def __group_by_day(self, df):
        df = df.copy()
        indexes = []
        flag = False
        day_number = []
        week_numbers = []
        month_number = []
        holiday_period = []
        number_of_flights = []
        number_of_destinations = []
        number_of_chartered = []
        number_of_sectors = []
        if 'NumberOfPassengers' in np.asarray(df.columns):
            flag = True
            mean_passenger_number = []
        years = np.array(df['Year'].unique())
        for year in years:
            year_df = df[df.Year == year]
            weeks = np.array(year_df.WeekNumber.unique())
            for week in weeks:
                week_df = year_df[year_df.WeekNumber == week]
                days = np.array(week_df.Day.unique())
                for day in days:
                    day_df = week_df[week_df.Day == day]
                    indexes.append(np.asarray(day_df.index))
                    day_number.append(day)
                    week_numbers.append(week)
                    month_number.append(np.array(day_df.Month.unique())[0])
                    holiday_period.append(
                        day_df.Holiday.value_counts().index[0])
                    number_of_flights.append(
                        len(np.asarray(day_df.FlightNumber.unique())))
                    number_of_destinations.append(
                        len(np.asarray(day_df.Destination.unique())))
                    number_of_chartered.append(
                        len(np.asarray(day_df[day_df.FlightType == 'C'])))
                    number_of_sectors.append(
                        len(np.asarray(day_df.Sector.unique())))
                    if flag:
                        mean_no_pass = np.mean(
                            day_df['NumberOfPassengers'].values)
                        mean_passenger_number.append(mean_no_pass)
        day_number = np.array([day_number]).T
        week_numbers = np.array([week_numbers]).T
        month_number = np.array([month_number]).T
        holiday_period = np.array([holiday_period]).T
        number_of_flights = np.array([number_of_flights]).T
        number_of_destinations = np.array([number_of_destinations]).T
        number_of_chartered = np.array([number_of_chartered]).T
        number_of_sectors = np.array([number_of_sectors]).T
        if flag:
            mean_passenger_number = np.array([mean_passenger_number]).T
            data = np.concatenate([day_number, week_numbers, month_number, holiday_period, number_of_flights,
                                   number_of_destinations, number_of_chartered, number_of_sectors, mean_passenger_number], axis=1)
            df = pd.DataFrame(data=data, columns=[
                'Day', 'WeekNumber', 'Month', 'Holiday', 'Flights', 'Destinations', 'Chartered', 'Sectors', 'NumberOfPassengers'])
            return df, indexes

        data = np.concatenate([day_number, week_numbers, month_number, holiday_period, number_of_flights,
                               number_of_destinations, number_of_chartered, number_of_sectors], axis=1)
        df = pd.DataFrame(data=data, columns=[
            'Day', 'WeekNumber', 'Month', 'Holiday', 'Flights', 'Destinations', 'Chartered', 'Sectors'])
        return df, indexes

    def __group_by_hour(self, df):
        df = df.copy()
        indexes = []
        flag = False
        hours = []
        day_number = []
        week_numbers = []
        month_number = []
        holiday_period = []
        number_of_flights = []
        number_of_destinations = []
        number_of_chartered = []
        number_of_sectors = []
        if 'NumberOfPassengers' in np.asarray(df.columns):
            flag = True
            mean_passenger_number = []
        years = np.array(df['Year'].unique())
        for year in years:
            year_df = df[df.Year == year]
            weeks = np.array(year_df.WeekNumber.unique())
            for week in weeks:
                week_df = year_df[year_df.WeekNumber == week]
                days = np.array(week_df.Day.unique())
                for day in days:
                    day_df = week_df[week_df.Day == day]
                    hours_ = np.array(day_df.Hour.unique())
                    for hour in hours_:
                        hour_df = day_df[day_df.Hour == hour]
                        indexes.append(np.asarray(hour_df.index))
                        hours.append(hour)
                        day_number.append(day)
                        week_numbers.append(week)
                        month_number.append(
                            np.array(hour_df.Month.unique())[0])
                        holiday_period.append(
                            hour_df.Holiday.value_counts().index[0])
                        number_of_flights.append(
                            len(np.asarray(hour_df.FlightNumber.unique())))
                        number_of_destinations.append(
                            len(np.asarray(hour_df.Destination.unique())))
                        number_of_chartered.append(
                            len(np.asarray(hour_df[hour_df.FlightType == 'C'])))
                        number_of_sectors.append(
                            len(np.asarray(hour_df.Sector.unique())))
                        if flag:
                            mean_no_pass = np.mean(
                                hour_df['NumberOfPassengers'].values)
                            mean_passenger_number.append(mean_no_pass)
        hours = np.array([hours]).T
        day_number = np.array([day_number]).T
        week_numbers = np.array([week_numbers]).T
        month_number = np.array([month_number]).T
        holiday_period = np.array([holiday_period]).T
        number_of_flights = np.array([number_of_flights]).T
        number_of_destinations = np.array([number_of_destinations]).T
        number_of_chartered = np.array([number_of_chartered]).T
        number_of_sectors = np.array([number_of_sectors]).T
        if flag:
            mean_passenger_number = np.array([mean_passenger_number]).T
            data = np.concatenate([hours, day_number, week_numbers, month_number, holiday_period, number_of_flights,
                                   number_of_destinations, number_of_chartered, number_of_sectors, mean_passenger_number], axis=1)
            df = pd.DataFrame(data=data, columns=[
                'Hour', 'Day', 'WeekNumber', 'Month', 'Holiday', 'Flights', 'Destinations', 'Chartered', 'Sectors', 'NumberOfPassengers'])
            return df, indexes
        data = np.concatenate([hours, day_number, week_numbers, month_number, holiday_period, number_of_flights,
                               number_of_destinations, number_of_chartered, number_of_sectors], axis=1)
        df = pd.DataFrame(data=data, columns=[
            'Hour', 'Day', 'WeekNumber', 'Month', 'Holiday', 'Flights', 'Destinations', 'Chartered', 'Sectors'])
        return df, indexes

    def from_sampled_to_schedule_format(self, estimates_vector, X_schedule_format):
        df = X_schedule_format.copy(deep=True)
        df = df.sort_values(by='ScheduleTime')
        index = df.index
        _, indexes = self.transform(df)
        passengers_numbers = np.zeros(X_schedule_format.shape[0])
        i = 0
        j = 0
        if self.sample_by == 'hours':
            for i in range(len(np.asarray(estimates_vector))):
                j1 = len(indexes[i])
                passengers_numbers[j:j+j1] = np.asarray(
                    estimates_vector).ravel()[i]
                j += j1
            seat_capacities = np.asarray(df['SeatCapacity'])
            load_factors = np.array([passengers_numbers/seat_capacities]).T
            load_factors = pd.DataFrame(
                data=load_factors, columns=['LoadFactor'])
            load_factors = load_factors.set_index(index)
            schedule = pd.concat([df, load_factors], axis=1)
            schedule = schedule.sort_index()
            return schedule

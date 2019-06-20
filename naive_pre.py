import pandas as pd
import csv
import numpy as np
from datetime import timedelta

def import_csv(filename: str) -> (pd.DataFrame, str):

    df = pd.read_csv(filename, index_col=None, quoting=csv.QUOTE_ALL, encoding="ISO-8859-1")

    df.columns = df.columns.str.strip()  # Remove whitespace.

    try:
        df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], format='%d-%m-%Y %H:%M:%S.%f')
    except ValueError:
        df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], format='%Y-%m-%d %H:%M:%S')

    keys = df.select_dtypes(include=['object'])
    for key in keys:
        df[key] = df[key].str.lower()

    df.sort_values(by=['event time:timestamp'], inplace=True)

    return df


def get_frequent_events(df: pd.DataFrame) -> (iter, iter):
    first_n = df.head(int(len(df) * 0.50))
    max_time = first_n['event time:timestamp'].max()
    last_events = df.loc[df.groupby('case concept:name')['event time:timestamp'].idxmax()]
    last_before = last_events[last_events['event time:timestamp'] <= max_time]
    chosen_events = df['case concept:name'].isin(last_before['case concept:name'].unique())
    df = df.loc[chosen_events]

    last_events = df.loc[df.groupby('case concept:name')['event time:timestamp'].idxmax()]
    frequent_events = []
    limit = last_events['event concept:name'].value_counts().mean()
    for event in range(0, last_events['event concept:name'].value_counts().size):
        if last_events['event concept:name'].value_counts()[event] >= limit:
            frequent_events.append(last_events['event concept:name'].value_counts().index[event])

    lifecycle_transition_at_end = None
    if 'event lifecycle:transition' in df.keys():
        lifecycle_transition_at_end = last_events['event lifecycle:transition'].value_counts().index[0]

    return frequent_events, lifecycle_transition_at_end


def aggregate(training_df: pd.DataFrame, prediction_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    print('Determining whether cases are finished ...')

    frequent_events, lifecycle_transition_at_end = get_frequent_events(training_df)

    def is_finished(row: pd.Series) -> bool:
        """Given a row, check whether a certain case is finished"""
        if lifecycle_transition_at_end:
            return (row['event concept:name'] in frequent_events and
                    row['event lifecycle:transition'] == lifecycle_transition_at_end)
        else:
            return row['event concept:name'] in frequent_events

    training_df['complete bool'] = training_df.apply(is_finished, axis=1)
    prediction_df['complete bool'] = prediction_df.apply(is_finished, axis=1)

    # Correcting false positive final events of cases for training data
    complete_mask = training_df['complete bool']
    complete_events = training_df[complete_mask]
    duplicated_mask = complete_events['case concept:name'].duplicated()
    duplicate_events = complete_events[duplicated_mask].copy()
    duplicate_events = duplicate_events.sort_values(by=['case concept:name', 'event time:timestamp'], inplace=False)
    duplicate_events.reset_index()
    limit = len(duplicate_events) - 2
    i = 0
    while i < limit:
        if duplicate_events.iloc[i]['case concept:name'] == duplicate_events.iloc[i + 1]['complete bool']:
            mask = training_df['eventID'] == duplicate_events.iloc[i]['eventID']
            training_df.loc[mask, 'complete bool'] = False
        i = i + 1

    return training_df, prediction_df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    bad_cases = df.loc[df.duplicated()]['case concept:name'].unique()
    for case in bad_cases:
        df = df[df['case concept:name'] != case]
    return df


def remove_null(df: pd.DataFrame) -> pd.DataFrame:
    bad_cases = df.loc[df.isnull().any(axis=1)]['case concept:name']
    for case in bad_cases:
        df = df[df['case concept:name'] != case]
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    keys = df.select_dtypes(include=numerics)
    for key in keys:
        df['difference'] = np.abs(df[key] - df[key].mean())
        threshold = df[key].std() * 3
        df['outlier'] = df['difference'] > threshold
        bad_cases = df[df['outlier']]['case concept:name'].unique()
        for case in bad_cases:
            df = df[df['case concept:name'] != case]

    df.drop(columns=['difference', 'outlier'], inplace=True)
    return df


def remove_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    complete_cases = df[df['complete bool']]['case concept:name'].unique()
    df = df[df['case concept:name'].isin(complete_cases)]

    return df


def clean_df(df: pd.DataFrame, outliers: bool) -> pd.DataFrame:
    count_before = df.shape[0]

    print('Removing duplicates ...')
    df = remove_duplicates(df)
    count_after = df.shape[0]
    print('' + str(count_before - count_after) + ' out of ' + str(
        count_before) + ' were removed during cleaning!')
    count_before = df.shape[0]

    print('Removing null values ...')
    df = remove_null(df)
    count_after = df.shape[0]
    print('' + str(count_before - count_after) + ' out of ' + str(
        count_before) + ' were removed during cleaning!')
    count_before = df.shape[0]

    if outliers:
        print('Removing outliers ...')
        df = remove_outliers(df)
        count_after = df.shape[0]
        print('' + str(count_before - count_after) + ' out of ' + str(
            count_before) + ' were removed during cleaning!')
        count_before = df.shape[0]

    print('Removing incomplete cases ...')
    df = remove_incomplete(df)
    count_after = df.shape[0]
    print('' + str(count_before - count_after) + ' out of ' + str(
        count_before) + ' were removed during cleaning!')
    return df

from datetime import timedelta
from collections import defaultdict

# duration: the duration from the start of the case til this event
# duration_remaining: the duration from the previous event til this event
# sequence: chain of events of the same case up to and including this event
# complete: whether this is the last event of a case that is finished


def aggregate_df(df: pd.DataFrame) -> pd.DataFrame:
    print('Converting event timestamps ...')
    df['event time:seconds'] = df['event time:timestamp'].apply(
        lambda x: timedelta(seconds=x.timestamp()).total_seconds())

    print('Storing shortcuts ...')

    # Write time of final event of each case as attribute
    df['final_event time:timestamp'] = np.NaN
    cases = df['case concept:name'].unique()
    for case in cases:
        case_mask = df['case concept:name'] == case
        case_events = df[case_mask]['event time:timestamp']
        start_time = case_events.min()
        final_time = case_events.max()
        df.loc[case_mask, 'start_event time:timestamp'] = start_time
        df.loc[case_mask, 'final_event time:timestamp'] = final_time

    df['final_event time:seconds'] = df['final_event time:timestamp'].apply(
        lambda x: timedelta(seconds=x.timestamp()).total_seconds())

    print('' + 'Calculating case durations up to event ...')

    df['duration time:timedelta'] = np.NaN

    def get_duration(row_in: pd.Series) -> timedelta:
        return row_in['event time:timestamp'] - row_in['start_event time:timestamp']

    df['duration time:timedelta'] = df.apply(get_duration, axis=1)
    df['duration time:seconds'] = df['duration time:timedelta'].apply(lambda x: x.total_seconds())

    print('Calculating event durations until end of case ...')

    df['duration_remaining time:timedelta'] = np.NaN

    def get_duration(row_in: pd.Series) -> timedelta:
        return row_in['final_event time:timestamp'] - row_in['event time:timestamp']

    df['duration_remaining time:timedelta'] = df.apply(get_duration, axis=1)
    df['duration_remaining time:seconds'] = df['duration_remaining time:timedelta'].apply(lambda x: x.total_seconds())

    df['duration_remaining time:timedelta'] = timedelta()

    print('Calculating event sequences for events ...')

    short_sequence_dictionary = dict()
    unique_events = df['event concept:name'].unique()
    counter = 0
    for event in unique_events:
        short_sequence_dictionary[event] = counter
        counter = counter + 1

    df.sort_values(by=['case concept:name', 'event time:timestamp'], inplace=True)

    sequence = ""
    last_case = ""
    column = []

    for n, row in df.iterrows():
        if last_case == row['case concept:name']:
            sequence = sequence + "+" + str(short_sequence_dictionary[row['event concept:name']])
        else:
            sequence = str(short_sequence_dictionary[row['event concept:name']])
            last_case = row['case concept:name']
        column.append(sequence)

    df['sequence concept:name'] = pd.Series(column).astype('str')

    df.sort_values(by=['event time:timestamp'], inplace=True)

    return df
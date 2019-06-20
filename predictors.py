import datetime
import time
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from ols_final import OLS_Predictor as ols
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

MIN_SAMPLES = 30
TAKE_COMPLETE_INTO_ACCOUNT = True

PREDICT_START = '' + "Starting {type} estimator at {time}"
PREDICT_DONE = '' + "Creating the {type} estimator took {secs} seconds."

def predict_mean(training_df: pd.DataFrame, prediction_df: pd.DataFrame, timing=True) -> pd.Series:
    """Perform a mean naive estimator on the prediction data based on the training data"""
    start_time = time.time()  # Time the duration of the prediction process.
    if timing:
        print(PREDICT_START.format(type="naive mean", time=str(datetime.datetime.now())))

    if len(training_df) < MIN_SAMPLES:
        return pd.Series([np.NaN for sample in range(0, len(prediction_df))])

    duration_remaining_sum = 0
    duration_remaining_count = 0

    column = []

    df = pd.concat([training_df, prediction_df], sort=False)

    training_mask = df['df_type'] == 2
    df.loc[training_mask, 'event time:timestamp'] = df.loc[training_mask, 'final_event time:timestamp']

    df.sort_values(by=['event time:timestamp', 'df_type'], inplace=True)

    for n, row in df.iterrows():
        if row['df_type'] == 1:
            duration_remaining_sum = duration_remaining_sum + row['duration_remaining time:seconds']
            duration_remaining_count = duration_remaining_count + 1
        if row['df_type'] == 2:
            if TAKE_COMPLETE_INTO_ACCOUNT and row['complete bool']:
                column.append(0)
            elif duration_remaining_count > MIN_SAMPLES:
                prediction = max(
                    (duration_remaining_sum / duration_remaining_count) - row['duration time:seconds'], 0)
                column.append(prediction)
            else:
                column.append(np.NaN)

    if timing:
        print(PREDICT_DONE.format(type="naive mean", secs=time.time() - start_time))
    return pd.Series(column, dtype='float')


def KNNpredict(Df, df_tesT, output, k=7):
    df_test_output = df_tesT.copy()
    #Create two lists, one with the variables where we will train on (since they have the same values for both testsets)
    #Create another list where the values in the columns do not allign, so we know which ones to drop.
    dummy_vars = list(Df)[1:20]
    dummy_var_lst = []
    lst_drop = []
    for i in dummy_vars:
        if set(Df[i].unique()) == set(df_tesT[i].unique()):
            dummy_var_lst.append(i)
        else:
            lst_drop.append(i)


    #Sort the dataframe for every case concept name on event time:timestamp
    df = Df.sort_values(['case concept:name', 'event time:timestamp'], ascending=[True, True])
    df_test = df_tesT.sort_values(['case concept:name', 'event time:timestamp'], ascending=[True, True])

    df_test.drop(['Naive Predictor'], axis=1, inplace=True)

    df['time_remaining_hours'] = [float(i.total_seconds()/86400) for i in df['remaining time']]
    df['time passed'] = [float(i.total_seconds()/86400) for i in df['time passed']]
    df_test['time passed'] = [float(i.total_seconds()/86400) for i in df_test['time passed']]

    #Create different date values so we can loop over each month and train the KNN per month
    df['date'] = df['event time:timestamp'].map(lambda x: 100*x.year + x.month)
    df_test['date'] = df_test['event time:timestamp'].map(lambda x: 100*x.year + x.month)
    
    #Sort the df's by date
    df['date'] = df.groupby(['case concept:name'])['date'].transform(max)
    df_test['date'] = df_test.groupby(['case concept:name'])['date'].transform(max)


    #Append other variables which should be dropped to the drop list
    lst_drop_test = lst_drop.copy()
    lst_drop.extend(['time_remaining_hours', 'date', 'remaining time', 'event time:timestamp', 'duration'])
    lst_drop_test.extend(['date', 'event time:timestamp', 'duration', 'remaining time'])
    
    
    #Create two lists of the month/years to loop over
    months_train = list(df['date'].unique())
    months_test = list(df_test['date'].unique())
    
    #Copy the dataframe and create dummies for the variables
    df_train = df.copy()
    df_train_dum = pd.get_dummies(df_train, columns=dummy_var_lst)
    df_test_dum = pd.get_dummies(df_test, columns=dummy_var_lst)
    
    #Create dummy column for the predictions of the KNN
    df_test_output['KNN'] = np.nan
    
    list_months = []
    months_list = months_train + list(set(months_test) - set(months_train))
    months_list.sort()

    count = 0
    print(len(months_list))
    print('KNN: entering for loop')
    for i in months_list:
        if count >= 11:
            continue

        list_months.append(i)
        dummy_df_train = df_train_dum[df_train_dum['date'].isin(list_months)]
        dummy_df_test = df_test_dum[df_test['date'].isin(list_months)]

        train_y = dummy_df_train['time_remaining_hours']

        dummy_df_train.drop(lst_drop, axis=1, inplace=True)
        dummy_df_test.drop(lst_drop_test, axis=1, inplace=True)
        train_x = dummy_df_train.copy()
        test_x = dummy_df_test.copy()
        try:
            train_x.drop('event concept:name', axis=1, inplace=True)
            test_x.drop('event concept:name', axis=1, inplace=True)
        except:
            pass
        KNR = KNeighborsRegressor(k)
        KNR.fit(train_x, train_y)
        if i in months_test:
            test_x['KNN'] = KNR.predict(test_x)
            df_test_output.update(test_x, overwrite=False)
        count += 1
        print(count)



    # put the result in the output queue
    output.put(df_test_output)
    # return df_test_output

def OLS_Predictor(train_df, test_df, output):
    # put the result in the output queue
    output.put(ols(train_df, test_df))
    # return ols(train_df, test_df)
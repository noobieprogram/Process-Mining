import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import time

def SplitTestOnTime(test_df, k):
    timespan = test_df.iloc[-1]['event time:timestamp'] - test_df.iloc[0]['event time:timestamp']
    dur = timespan / k # duration of a single chunk
    start = test_df.iloc[0]['event time:timestamp']
    old = start - dur
    chunks = []
    for i in range(5):
        if i == 0:
            rows = test_df.loc[(test_df['event time:timestamp'] > old) & (test_df['event time:timestamp'] <= start + dur*(i+1))]
            chunks.append(rows)
        else:
            rows = test_df.loc[(test_df['event time:timestamp'] > start + dur*i) & (test_df['event time:timestamp'] <= start + dur*(i+1))]
            chunks.append(rows)
    return chunks


def SplitTrainSet(test_df, train_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    chunks = SplitTestOnTime(test_df, 5)
        
    # add first timestamps in the test chunks into a list
    first_in_chunk = []
    for chunk in chunks:
        first_in_chunk.append(chunk.iloc[0]['event time:timestamp'])
    
    # sort the train data set on time and reset the index
    train_df.sort_values(by=['event time:timestamp'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)
      
    # make a list to store row indices
    train_buckets_index = [[] for i in range(len(first_in_chunk))]
    
    # add row indices into respective buckets     
    for i, first in enumerate(first_in_chunk):
        for index, row in train_df.iterrows():
            if row['event time:timestamp'] < first:
                train_buckets_index[i].append(index)
                
            else:
                break
        # print('done with chunk {}'.format(i))
                
    # add rows from train set based on the indeces
    # unless the indices contain less than 1% of the training set
    treshold = len(train_df)/100 # 1% of the training set
    train_buckets = {}
    for i, indices in enumerate(train_buckets_index):
        if len(train_buckets_index[i]) > treshold:
            train_buckets[i] = train_df.iloc[indices]
        else:
            pass    
    
    return chunks, train_buckets


def OLS_Predictor(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # add variables of required type into the dataframes
    train_df['rem_time_days'] = [float(i.total_seconds()/(3600*24)) for i in train_df['remaining time']]
    train_df['time_passed_days'] = [float(i.total_seconds()/(3600*24)) for i in train_df['time passed']]
    train_df['event Cumulative net worth (EUR)'] = train_df['event Cumulative net worth (EUR)'].astype(float)

    test_df['rem_time_days'] = [float(i.total_seconds()/(3600*24)) for i in test_df['remaining time']]
    test_df['time_passed_days'] = [float(i.total_seconds()/(3600*24)) for i in test_df['time passed']]
    test_df['event Cumulative net worth (EUR)'] = test_df['event Cumulative net worth (EUR)'].astype(float)
    
    # prepare variables from the datasets for OLS model
    
    # list of numeric variables
    variables = ['event Cumulative net worth (EUR)', 'time_passed_days']
    squared = ['event Cumulative net worth (EUR)']
    
    s1 = time.time()
    # add squared variables to df's and list of variables
    sq_var = []
    for var in squared:
        train_df[str(var) + ' sq'] = train_df[var] ** 2
        test_df[str(var) + ' sq'] = test_df[var] ** 2
        sq_var.append(str(var) + ' sq')
    
    variables = variables + sq_var
    
    # list of categorical vars to be made into dummies
    cat = ['case Item Category', 'case Spend area text', 'case Company', 
           'case Document Type', 'case Spend classification text']
    
    # get the dummies
    dummy_train = pd.get_dummies(train_df[cat])
    dummy_test = pd.get_dummies(test_df[cat])
    
    # add dummy columns to test and train so both df's have the same columns
    only_test = [] # contains columns in test but not in train
    only_train = [] # contains columns in train but not in test
    
    for column in dummy_train:
        if column not in dummy_test.columns.values:
            only_train.append(column)

    for column in dummy_test:
        if column not in dummy_train.columns.values:
            only_test.append(column)    
    
    # set new columns to 0
    for column in only_test:
        dummy_train[str(column)] = 0

    for column in only_train:
        dummy_test[str(column)] = 0
    
    # add dummy columns to original df's
    train_df = pd.concat([train_df, dummy_train], axis = 1)
    test_df = pd.concat([test_df, dummy_test], axis = 1)
    
    # list all dummy columns and add them to variables
    dummy_cols = list(dummy_train.columns.values)
    variables = variables + dummy_cols
    e1 = time.time()
    # print('Done with setting up variables. It took {:.3f} sec'.format(e1-s1))
    
    # make chunk of the test set and respective buckets from the train set
    # both with the use of some helper functions
    # print('Start splitting test and train set')
    s2 = time.time()
    test_chunks, train_buckets = SplitTrainSet(test_df, train_df)
    e2 = time.time()
    print('OLS: done with splitting test and train set. It took {:.3f} sec'.format(e2-s2))
    
    # make OLS models for each bucket
    # print('Start to make models')
    models = {}

    s3 = time.time()

    for key, bucket in train_buckets.items():
        models[key] = LinearRegression().fit(bucket[variables], bucket['rem_time_days'])
        # print('R^2: ', models[key].score(bucket[variables], bucket['rem_time_days']))

    e3 = time.time()
    # print('Making models took {:.3f}s'.format(e3-s3))
    
    # make predictions for each test chunk
    # print('Start with predictions')
    
    s4 = time.time()
    for i, chunk in enumerate(test_chunks):
        if i in models.keys():
            chunk.loc[:, 'OLS_pred'] = models[i].predict(chunk[variables])
        else:
            chunk.loc[:, 'OLS_pred'] = np.nan
    
    # replace negative predictions with zeros
    for chunk in test_chunks:
        chunk.loc[chunk['OLS_pred'] < 0, 'OLS_pred'] = 0
    
    # concatinate all chucnks together
    result = pd.concat(test_chunks)
    
    # add lists of columns to remove and remove them
    remove = dummy_cols + ['event Cumulative net worth (EUR) sq', 'rem_time_days', 'time_passed_days']
    result.drop(columns = remove, inplace = True)
    
    e4 = time.time()
    print('OLS: done with predictions. It took {:.3f} sec.'.format(e4-s4))
    
    return result
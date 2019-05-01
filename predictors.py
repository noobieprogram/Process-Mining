import datetime
import pandas as pd
from statsmodels.formula.api import ols
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def naivePredict(Test, linked_training):
    starttimes = {}
    i= 0
    skipcount=0
    currenttime = linked_training[i][-1]['event time:timestamp']
    test = Test.copy()
    totaltime = datetime.timedelta(0)
    for event in test:
        #creating the proper durationlist
        while event['event time:timestamp'] > currenttime and i < len(linked_training):
            totaltime += linked_training[i][0]['duration']
            i+=1
            if i < len(linked_training):
                currenttime =  linked_training[i][-1]['event time:timestamp']
            average = totaltime/i
        if i > 19:
            if event['case concept:name'] in starttimes:
                estimate = average - (event['remaining time'])
                if estimate.total_seconds() < 0:
                    event['Naive Predictor'] = datetime.timedelta(0)
                else:
                    event['Naive Predictor'] = estimate
            else:
                event['Naive Predictor'] = average
                starttimes[event['case concept:name']] = event['event time:timestamp']
        else:
            event['Naive Predictor'] = datetime.timedelta(-10)
            skipcount+=1
    if skipcount > 0:
        print('skipcount is', skipcount)
    return test

def olsPredict(test, linkedtrain):
    output = test.copy()
    i =0
    trainingdict = {'hours_remaining':[],'hours_passed':[],'event_worth':[],'event_concept':[],'company':[]}
    predictions = []
    for event in output:
        while i< len(linkedtrain):
            if event['event time:timestamp'] > linkedtrain[i][-1]['event time:timestamp']:
                for case in linkedtrain[i]:
                    trainingdict['hours_remaining'].append(case['remaining time'].total_seconds()/3600)
                    trainingdict['hours_passed'].append(case['time passed'].total_seconds()/3600)
                    trainingdict['event_worth'].append(float(case['event Cumulative net worth (EUR)']))
                    trainingdict['event_concept'].append(case['event concept:name'])
                    trainingdict['company'].append(case['case Company'])
                i += 1
            else:
                break
            if i >= len(linkedtrain):
                linreg = ols("hours_remaining ~ hours_passed", trainingdict).fit()
                break
            elif not event['event time:timestamp'] > linkedtrain[i][-1]['event time:timestamp']:
                linreg = ols("hours_remaining ~ hours_passed", trainingdict).fit()

        if i > 19:
            currenttest = {'hours_passed':event['time passed'].total_seconds()/3600,
                           'hours_remaining':event['remaining time'].total_seconds()/3600
                           ,'event_worth':float(event['event Cumulative net worth (EUR)'])
                           ,'event_concept':event['event concept:name']
                           ,'company':event['case Company']
                          }
            pred_val = linreg.predict(pd.DataFrame(currenttest, index=[0]))
            predictions.append(pred_val)
        else:
            predictions.append(-10)
    output = olsPredict2(predictions, test)
    return output

def olsPredict2(predic1, test):
    pred1 = predic1.copy()
    output = test.copy()
    for i in range(len(pred1)):
        if not type(pred1[i]) == type(0):
            if pred1[i][0] < 0:
                pred1[i] = 0
        output[i].update({'OLS':pred1[i]})
    for i in output:
        i['OLS'] = int(i['OLS'])
    return output

def KNNpredict(Df, df_tesT, k=7):
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

    df_test.drop(['Naive Predictor', 'OLS'], axis=1, inplace=True)

    df['time_remaining_hours'] = [float(i.total_seconds()/3600) for i in df['remaining time']]
    df['time passed'] = [float(i.total_seconds()/3600) for i in df['time passed']]
    df_test['time passed'] = [float(i.total_seconds()/3600) for i in df_test['time passed']]

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

                             
    for i in months_list:
        list_months.append(i)
        dummy_df_train = df_train_dum[df_train_dum['date'].isin(list_months)]
        dummy_df_test = df_test_dum[df_test['date'].isin(list_months)]

        train_y = dummy_df_train['time_remaining_hours']

        dummy_df_train.drop(lst_drop, axis=1, inplace=True)
        dummy_df_test.drop(lst_drop_test, axis=1, inplace=True)
        train_x = dummy_df_train.copy()
        test_x = dummy_df_test.copy()
        train_x.drop('event concept:name', axis=1, inplace=True)
        test_x.drop('event concept:name', axis=1, inplace=True)
        KNR = KNeighborsRegressor(k)
        KNR.fit(train_x, train_y)
        if i in months_test:

            test_x['KNN'] = KNR.predict(test_x)
            df_test_output.update(test_x, overwrite=False)
    return df_test_output
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# mean squared error
def calcMSE(dataframe, predictor):
    true = []
    estimate = []
    for event in dataframe.iterrows():

        if event[1]['remaining time'].total_seconds()/86400 < 0:
            continue
        true.append(event[1]['remaining time'].total_seconds()/86400)
        try:
            estimate.append(event[1][predictor].total_seconds()/86400)
        except:
            estimate.append(event[1][predictor]/86400)
    # plot(true, estimate, dataframe['time passed'])
    return mean_squared_error(true, estimate)

def plot(true, estimate, y):
    plt.scatter(true, color = 'green', alpha=0.2, y = y)
    plt.scatter(estimate, color = 'red', alpha = 0.2, y=y)
    # ax1.plot(ols_res1['time_passed_days'].iloc[44:], ols_res1['OLS_pred'].iloc[44:], color='g')
    plt.legend(loc='upper right');
    plt.show()

# standard error
def calcError(dataframe, predictor):
    error_list = []
    for event in dataframe.iterrows():
        remtime = event[1]['remaining time'].total_seconds()/86400
        try:
            error = (remtime - event[1][predictor].total_seconds()/86400)
        except:
            error = (remtime - event[1][predictor])
        error_list.append(error)
    return error_list
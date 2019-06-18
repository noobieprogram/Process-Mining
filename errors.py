import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# mean squared error
def calcMSE(dataframe, predictor):
    true = []
    estimate = []
    for event in dataframe.iterrows():
            true.append(event[1]['remaining time'].total_seconds()/86400)
            try:
                estimate.append(event[1][predictor].total_seconds()/86400)
            except:
                estimate.append(event[1][predictor]/86400)

    return mean_squared_error(true, estimate)


# standard error
def calcError(dataframe, predictor):
    error_list = []
    for event in dataframe.iterrows():
        remtime = event[1]['remaining time'].total_seconds()/3600
        try:
            error = (remtime - event[1][predictor].total_seconds()/3600)
        except:
            error = (remtime - event[1][predictor])
        error_list.append(error)
    return error_list
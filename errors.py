import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calcMSE(dataframe, predictorlist):
    sqe_list = [[] for i in predictorlist]
    for event in dataframe.iterrows():
        for i in range(len(predictorlist)):
            remtime = event[1]['remaining time'].total_seconds()/3600
            try:
                sqerror = (remtime - event[1][predictorlist[i]].total_seconds()/3600)**2
            except:
                sqerror = (remtime - event[1][predictorlist[i]]/3600)**2
            sqe_list[i].append(sqerror)
    mselist = []
    length = len(sqe_list[1])
    for i in range(len(predictorlist)):
        mse = sum(sqe_list[i])/length
        mselist.append(mse)
    return(mselist)


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


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import datetime
from errors import calcError

def plotEstimate(test, testfile, estimator):

    TrueRemaining = [case.total_seconds() / 3600 for case in test['remaining time']]
    TimePassed = [case.total_seconds() / 3600 for case in test['time passed']]
    try:
        EstiRemaining = [case.total_seconds()/3600 for case in test[estimator]]
    except:
        EstiRemaining = [case for case in test[estimator]]
    for i in range(len(EstiRemaining)):
        if EstiRemaining[i] <0:
            EstiRemaining[i], TimePassed[i], TrueRemaining[i] = None, None, None
    try:
        while True:
            TrueRemaining.remove(None)
            EstiRemaining.remove(None)
            TimePassed.remove(None)
    except:
        pass
    plt.figure(figsize=(10, 10))
    plt.scatter(TimePassed, TrueRemaining, color='green', alpha=0.01)
    plt.scatter(TimePassed, EstiRemaining, color='red', alpha=0.01)
    estimate = mpatches.Patch(color='red', label=estimator)
    true = mpatches.Patch(color='green', label='True time remaining')
    plt.xlabel('time passed in hours')
    plt.ylabel('remaining time in hours')
    plt.legend(handles=[estimate, true])
    plt.title(estimator + ' from ' + testfile[:-9])
    plt.savefig(testfile[:-8] + estimator + '-Plot.png')

def plotSqError(test, estimator):
    #input voor de functie, test = lijst dictionaries, 

    TrueRemaining =[event.total_seconds() / 3600 for event in test['remaining time']]
    try:
        EstiRemaining = [case.total_seconds()/3600 for case in test[estimator]]
    except:
        EstiRemaining = [case for case in test[estimator]]



    try:
        sqe_list = []
        for evenT in test.iterrows():
            event = evenT[1]
            remtime = event['remaining time'].total_seconds()/3600
            sqerror = (remtime - event[estimator].total_seconds()/3600)**2
            sqe_list.append(sqerror)
    except:
        sqe_list = []
        for evenT in test.iterrows():
            event = evenT[1]
            remtime = event['remaining time'].total_seconds()/3600
            sqerror = (remtime - event[estimator])**2
            sqe_list.append(sqerror)
    for i in range(len(EstiRemaining)):
        if EstiRemaining[i] <0:
            EstiRemaining[i], TrueRemaining[i], sqe_list[i] = None, None, None
    try:
        while True:
            TrueRemaining.remove(None)
            sqe_list.remove(None)
            EstiRemaining.remove(None)
    except:
        pass
    plt.figure(figsize=(10, 10))
    plt.scatter(TrueRemaining, sqe_list, color='green', alpha=0.01)
    plt.xlabel('True Time Remaining in hours')
    plt.ylabel('Squared Error')

    plt.title(estimator)
    plt.savefig(estimator + ' squared error plot.png')

def errorDist(df, predictor):
    error = calcError(df, predictor)
    plt.hist(error, bins = 15)
    plt.xlabel("Error in hours")
    plt.ylabel("Count")
    plt.title("Error for " + predictor)
    plt.savefig(predictor + ' error distribution plot.png')
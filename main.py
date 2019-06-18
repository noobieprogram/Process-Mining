import csv
import datetime
import sys
import pandas as pd
from predictors import naivePredict, KNNpredict, OLS_Predictor 
from errors import calcMSE
import utilities as ut
from plotting import *

# entry to the code
def read():
    try:
        trainingfile = sys.argv[1]
        testfile = sys.argv[2]
        outputfile = sys.argv[3]
    except:
        print('Input was not given in the correct format')
        testfile ='/Users/abdullahsaeed/Documents/2IOI0-2/10%subset_2019-test.csv'
        trainingfile = '/Users/abdullahsaeed/Documents/2IOI0-2/10%subset_2019-training.csv'
        outputfile = 'output.csv'

    main(testfile, trainingfile, outputfile)


def main(testfile, trainingfile, outputfile):
    # Reading the data into a list of dictionaries
    starttime = datetime.datetime.today()

    # read files as list of dictionaries
    test = [dict(line) for line in csv.DictReader(open(testfile, 'r', encoding = "ISO-8859-1"))]
    training = [dict(line) for line in csv.DictReader(open(trainingfile, 'r', encoding = "ISO-8859-1"))]
    print('test set has', len(test),'instances, training set has', len(training),'instances')

    # Doing all preprocessing; 
    # cutting unfinished cases, 
    # appending true values and making a list, 
    # linked by cases
    # All lists are sorted on time
    training, linked_training = ut.preProcess(training)
    test, linked_test = ut.preProcess(test)
    print('All preprocessing has been done')

    # call average predictor
    startnaive = datetime.datetime.today()
    test = naivePredict(test, linked_training)

    df_training = ut.dictToDf(training)
    df_test = ut.dictToDf(test)


    # KNN algorithm
    # startKNN = datetime.datetime.today()
    df_test = KNNpredict(df_training, df_test)
    # print('KNN finished in', datetime.datetime.today() - startKNN)
    
    df_test = OLS_Predictor(df_training, df_test)
    # plotEstimate(testdf, testfile, 'KNN')
    # print("KNN plot made")

    # # Printing the MSEs of all estimators
    # eslst = ['Naive Predictor', 'OLS', 'KNN']

    # for i in range(len(eslst)):
    #     print(eslst[i], 'has a mean squared error of', MSEs[i])
    #     plotEstimate(df_test, testfile, eslst[i])
    #     plotSqError(df_test, eslst[i])
    #     errorDist(df_test, eslst[i])

    df_test.to_csv(outputfile)
    # print('the entire program took', datetime.datetime.today() - starttime)

# Call the function
read()
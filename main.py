import csv
import time
import sys
import pandas as pd
from predictors import naivePredict, KNNpredict, OLS_Predictor
from errors import calcMSE
import utilities as ut
from plotting import *
from multiprocessing import Process, Queue
import warnings

# ignore warnings thrown by libraries
warnings.filterwarnings('ignore')

def main(test, training, outputfile):

    # converting files into list of dictionaries, using appropriate encoding
    test = [dict(line) for line in csv.DictReader(open(testfile, 'r', encoding="ISO-8859-1"))]
    training = [dict(line) for line in csv.DictReader(open(trainingfile, 'r', encoding="ISO-8859-1"))]
    # just initial output to confirm that everything has been read properly
    print('Input has been read correctly')
    print('Training set has', len(training), 'instances;', 'test set has', len(test), 'instances')

    # do pre-processing: cutting unfinished cases, link by cases, sort lists on time
    training, linked_training = ut.preProcess(training)
    test, linked_test = ut.preProcess(test)
    print('All preprocessing has been done')

    # Naive estimator
    print("Naive predictor has started")
    test = naivePredict(test, linked_training)
    print("Naive predictor has ended")

    # convert to Dataframes for OLS and KNN
    df_training = ut.dictToDf(training)
    df_test = ut.dictToDf(test)

    # queue for output of KNN and OLS
    out = Queue(2)
    args = [df_training, df_test, out]

    # create and start new processes for KNN and OLS
    p1 = Process(target  = KNNpredict, args = args)
    p2 = Process(target = OLS_Predictor, args = args)
    print("KNN and OLS predictors have started in parallel")
    p1.start()
    p2.start()

    # KNN usually finishes first, so this is likely to be KNN
    first = out.get()
    # drop unnecessary columns
    first.drop(first.columns[0], axis = 1, inplace = True)

    second = out.get()
    # drop unnecessary columns
    second.drop(second.columns[0], axis = 1, inplace = True)

    final =  pd.merge(first, second, on='eventID ', suffixes=('KNN', 'OLS'))
    # drop unnecessary columns
    final.drop(final.columns[27: -1], axis = 1, inplace = True)
    # write final output to file
    final.to_csv(outputfile)

    # first.to_csv("output2.csv")
    # second.to_csv("output3.csv")

    # finish the processes
    p1.join()
    p2.join()
    p1.terminate()
    p2.terminate()
    print("All predictors have finished, program will terminate now!")

# entry point to the program
if __name__ == '__main__':
    start = time.clock()
    try:
        trainingfile = sys.argv[1]
        testfile = sys.argv[2]
        outputfile = sys.argv[3]
    except: # useful for development/debuggings
        print('Input was not given in the correct format')
        testfile ='10%subset_2019-test.csv'
        trainingfile = '10%subset_2019-training.csv'
        outputfile = 'output.csv'

    # just a fancy intro
    print('''
    $$$$$$$\                                                                    $$\      $$\ $$\           $$\                     
    $$  __$$\                                                                   $$$\    $$$ |\__|          \__|                    
    $$ |  $$ | $$$$$$\   $$$$$$\   $$$$$$$\  $$$$$$\   $$$$$$$\  $$$$$$$\       $$$$\  $$$$ |$$\ $$$$$$$\  $$\ $$$$$$$\   $$$$$$\  
    $$$$$$$  |$$  __$$\ $$  __$$\ $$  _____|$$  __$$\ $$  _____|$$  _____|      $$\$$\$$ $$ |$$ |$$  __$$\ $$ |$$  __$$\ $$  __$$\ 
    $$  ____/ $$ |  \__|$$ /  $$ |$$ /      $$$$$$$$ |\$$$$$$\  \$$$$$$\        $$ \$$$  $$ |$$ |$$ |  $$ |$$ |$$ |  $$ |$$ /  $$ |
    $$ |      $$ |      $$ |  $$ |$$ |      $$   ____| \____$$\  \____$$\       $$ |\$  /$$ |$$ |$$ |  $$ |$$ |$$ |  $$ |$$ |  $$ |
    $$ |      $$ |      \$$$$$$  |\$$$$$$$\ \$$$$$$$\ $$$$$$$  |$$$$$$$  |      $$ | \_/ $$ |$$ |$$ |  $$ |$$ |$$ |  $$ |\$$$$$$$ |
    \__|      \__|       \______/  \_______| \_______|\_______/ \_______/       \__|     \__|\__|\__|  \__|\__|\__|  \__| \____$$ |
                                                                                                                         $$\   $$ |
                                                                                                                         \$$$$$$  |
                                                                                                                          \______/ 
    ''')
    main(testfile, trainingfile, outputfile)
    print("Time taken:", time.clock() - start)
    print(''' ✫彡 Done!''')
import csv
import time
import sys
from ols_gb import PrepareDatasets
from predictors import *
import utilities as ut
from plotting import *
import multiprocessing as mp


def main(testfile: str, trainingfile: str, outputfile: str):

    # converting files into list of dictionaries, using appropriate encoding
    test = [dict(line) for line in csv.DictReader(open(testfile, 'r', encoding="ISO-8859-1"))]
    training = [dict(line) for line in csv.DictReader(open(trainingfile, 'r', encoding="ISO-8859-1"))]
    # just initial output to confirm that everything has been read properly
    print('Datasets have been read correctly')
    print('Training set has', len(training), 'instances;', 'test set has', len(test), 'instances')

    # do pre-processing: cutting unfinished cases, link by cases, sort lists on time
    training, linked_training = ut.preProcess(training)
    test, linked_test = ut.preProcess(test)
    print('All pre-processing has been done')

    # Naive estimator
    print("Naive predictor has started")
    test = naivePredict(test, linked_training)
    print("Naive predictor has ended")

    # convert to Dataframes for OLS and KNN
    df_training = ut.dictToDf(training)
    df_test = ut.dictToDf(test)

    #do splitting of data for OLS and GB
    print("KNN, OLS and Gradient Boosting predictors have started in parallel")
    test_chunks, train_buckets, variables, dummy_cols = PrepareDatasets(df_training, df_test)

    mp.set_start_method('spawn')

    # queue for output of KNN, GB and OLS
    out = mp.Queue()
    args = [df_training, df_test, out]
    args_2 = [test_chunks, train_buckets, variables, dummy_cols, out]

    # create and start new processes for KNN, GB and OLS
    p1 = mp.Process(target = KNNpredict, args = args)
    p2 = mp.Process(target = ols, args = args_2)
    p3 = mp.Process(target = gradient, args = args_2)
    p1.start()
    p2.start()
    p3.start()

    # retrieve outputs
    first = out.get()
    second = out.get()
    third = out.get()

    # merge the outputs
    temp = pd.merge(df_test, first, on='eventID ', suffixes=('', ' '))
    temp2 = pd.merge(temp, second, on='eventID ', suffixes=('', ' '))
    final = pd.merge(temp2, third, on = 'eventID ', suffixes = ('', ' '))
    final.drop(final.columns[0], axis = 1, inplace = True)
    final.to_csv(outputfile)

    # finish the processes
    p1.join()
    p2.join()
    p3.join()
    p1.terminate()
    p2.terminate()
    p3.terminate()
    print("All predictors have finished, program will proceed to terminate now!")

# entry point to the program
if __name__ == '__main__':
    start = time.time()
    flag = False
    try:
        trainingfile = sys.argv[1]
        testfile = sys.argv[2]
        outputfile = sys.argv[3]
        flag = True
    except: # useful for development/debugging
        ut.fancyPrint()
        print('Input was not given in the correct format, therefore the default datasets will be loaded')
        testfile ='/10%subset_2019-test.csv'
        trainingfile = '/10%subset_2019-training.csv'
        outputfile = 'output.csv'

    # it's just a fancy intro text, nothing to worry about
    if flag == True:
        ut.fancyPrint()

    main(testfile, trainingfile, outputfile)
    # print total time taken
    print("Time taken:", str(time.time() - start)[0:6], 'seconds')
    print(''' ✫彡 Done!''')
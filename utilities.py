import pandas as pd
import datetime

def dictToDf(file):
    dfdict = {}
    for columnname in file[0].keys():
        dfdict[columnname] = [event[columnname] for event in file]
    df = pd.DataFrame(data=dfdict)
    return(df)

def preProcess(file):
    newfile = sorted(fixdate(file), key=lambda n: n['event time:timestamp'])
    linkedcases = linkedCaseSort(linkCases(newfile))
    newfile = appendTrue(linkedcases, newfile)
    linkedcases = linkedCaseSort(linkCases(newfile))
    newfile = unfinished(newfile, linkedcases)
    linkedcases = linkedCaseSort(linkCases(newfile))
    newfile, linkedcases = outliers(linkedcases)
    newfile = appendTrue(linkedcases, newfile)
    return (newfile, linkedcases)

# fixes date format in the files
def fixdate(file):
    file1 = file.copy()
    for i in file1:
        date = i['event time:timestamp']
        day, month, year, hour, minute, second = date[0:2], date[3:5], date[6:10], date[11:13],date[14:16],date[17:19]
        actualdate = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        i['event time:timestamp'] = actualdate
    return file1

def linkCases(file):
    Applications = [Dict['case concept:name'] for Dict in file]
    unique = set(Applications)
    linked_file = {number:[] for number in unique}

    for dictionary in file:
        linked_file[dictionary['case concept:name']].append(dictionary)
    return linked_file

def linkedCaseSort(linkedlist):
    endtimes = {i:linkedlist[i][-1]['event time:timestamp'] for i in linkedlist}
    endtimes = sorted(endtimes.items(), key=lambda k: k[1])
    goodorder = [i[0] for i in endtimes]
    newlist = []
    for i in goodorder:
        newlist.append(linkedlist[i])
    return newlist

def appendTrue(linked_file, file):
    output = file.copy()
    endtimes = {i[-1]['case concept:name']: i[-1]['event time:timestamp'] for i in linked_file}
    durations ={i[-1]['case concept:name']: i[-1]['event time:timestamp']-i[0]['event time:timestamp'] for i in linked_file}
    for case in output:
        case['duration'] = durations[case['case concept:name']]
        case['remaining time'] = (endtimes[case['case concept:name']] - case['event time:timestamp'])
    starttimes = {}
    for case in output:
        if case['case concept:name'] in starttimes:
            case['time passed'] = case['event time:timestamp'] - starttimes[case['case concept:name']]
        else:
            starttimes[case['case concept:name']] = case['event time:timestamp']
            case['time passed'] = datetime.timedelta(0)
    return(output)

# general case, with 90% threshold
def unfinished(file, linkedFile, includedpercentage = 90):
    includedamount = includedpercentage/100*len(file)
    endvalues = [i[-1]['event concept:name'] for i in linkedFile]
    unique = list(set(endvalues))
    occurrenceDict = {i:endvalues.count(i) for i in unique}
    occurrenceDict = {i[0]:i[1] for i in sorted(occurrenceDict.items(), key=lambda kv: kv[1], reverse=True)}
    endevents = []
    for i in occurrenceDict:
        endevents.append(i)
        includedamount -= occurrenceDict[i]
        if includedamount <=0:
            break
    output = []
    for case in linkedFile:
        if case[-1]['event concept:name'] in endevents:
            for event in case:
                output.append(event)
    output = sorted(output, key = lambda k: k['event time:timestamp'])
    return(output)

# cuts cases that are outliers
def outliers(linkedFile):
    linked_file = linkedFile.copy()
    outliers = [i if i[0]['duration'].total_seconds()/3600 > 60000 else 0 for i in linkedFile ]
    number = 0
    for outlier in outliers:
        if outlier != 0:
            i = 0
            fuckthis = True
            while fuckthis:
                outlier_i = outlier[i]['event time:timestamp']
                outlier_2 = outlier[i+1]['event time:timestamp']
                if (outlier_2 - outlier_i).total_seconds()/3600 > 20000:
                    outlier[i] = 0
                    outlier = [x for x in outlier if x != 0]
                    fuckthis = False
                else:
                    outlier[i] = 0
                    i+=1
            linked_file[number] = outlier
        number+=1
    file = []
    for case in linked_file:
        for event in case:
            file.append(event)
    file = sorted(file, key = lambda k: k['event time:timestamp'])
    return (file, linked_file)
# coding: utf-8

from random import randint
# import matplotlib

import numpy as np
import torch
import pickle
import regrFuncs as rF

def predict(v1, reg):
    probs = reg.predict_proba(v1)
    label = np.array([int(x) for x in reg.predict(v1).flatten()])
    return label, probs.tolist()

def runtests_not_batch(name, classifier, model, tasks, outpaths, label2id, testDATA):
    modelname = name + classifier
    print('\n\n'+'**'*40)
    print("\nRunning tests for {0}...\n".format(modelname))
    print('**'*40 + '\n')
    regressor = pickle.load(open('{0}{1}'.format(outpaths['REGR_MODEL_PATH'], modelname), 'rb'))
    for task in tasks:
        with open(('{0}X.'+testDATA+'.{1}').format(outpaths['TEST_DATA_PATH'], task)) as f:
            X = f.readlines()
        try:
            true = np.loadtxt(('{0}labels.'+testDATA+'.{1}').format(outpaths['TEST_DATA_PATH'], task), 
                dtype = int)
        except ValueError:
            with open(('{0}labels.'+testDATA+'.{1}').format(outpaths['TEST_DATA_PATH'], task)) as f:
                true = f.readlines()
                true = [label2id[x.strip().upper()] for x in true]

        
        embedA = rF.embed(model, X, 1, name)
        
        labels, confs = predict(embedA, regressor)      
              
        np.savetxt(('{0}'+testDATA+'.{1}_labels_{2}').format(outpaths['TEST_OUT_PATH'], task, modelname),
            labels,
            fmt = '%i')

        np.savetxt(('{0}'+testDATA+'.{1}_confs_{2}').format(outpaths['TEST_OUT_PATH'], task, modelname), 
            confs)
        
        rights = (labels == true)
        acc = sum(rights)*100.0/len(rights)
        
        counts = np.bincount(true, minlength = 3)*100.0/len(rights)
        counts = np.array([round(x,2) for x in counts])
        
        print("\nTaskDATA: {0}, Split: {1}, o: {2}, I: {3}".format(testDATA, task, counts[0], counts[1]))

        print("Accuracy: {0}".format(round(acc,3)))

    return


def runtests(name, classifier, model, tasks, outpaths, label2id, testDATA):
    modelname = name + classifier
    print('\n\n'+'**'*40)
    print("\nRunning tests for {0}...\n".format(modelname))
    print('**'*40 + '\n')
    regressor = pickle.load(open('{0}{1}'.format(outpaths['REGR_MODEL_PATH'], modelname), 'rb'))
    for task in tasks:
        rights = []
        with open(('{0}X.'+testDATA+'.{1}').format(outpaths['TEST_DATA_PATH'], task)) as f:
            X = f.readlines()
        try:
            true = np.loadtxt(('{0}labels.'+testDATA+'.{1}').format(outpaths['TEST_DATA_PATH'], task), 
                dtype = int)
        except ValueError:
            with open(('{0}labels.'+testDATA+'.{1}').format(outpaths['TEST_DATA_PATH'], task)) as f:
                true = f.readlines()
                true = [label2id[x.strip().upper()] for x in true]

        batch_size = 256
        for ii in range(0, len(true), batch_size):
            if ii == len(true) / batch_size* batch_size:
                X_batch = X[ii:]
                true_batch = true[ii:]
            else:
                X_batch = X[ii:ii + batch_size]
                true_batch = true[ii:ii + batch_size]
            embedA = rF.embed(model, X_batch, 1, name)
            
            labels, confs = predict(embedA, regressor)    

              
            np.savetxt(('{0}'+testDATA+'.{1}_labels_{2}').format(outpaths['TEST_OUT_PATH'], task, modelname),
                labels,
                fmt = '%i')

            np.savetxt(('{0}'+testDATA+'.{1}_confs_{2}').format(outpaths['TEST_OUT_PATH'], task, modelname), 
                confs)
            
            rights_batch = (labels == true_batch)
            rights.extend(rights_batch)

        acc = sum(rights)*100.0/len(rights)
        
        counts = np.bincount(true, minlength = 3)*100.0/len(rights)
        counts = np.array([round(x,2) for x in counts])
        
        print("\nTaskDATA: {0}, Split: {1}, o: {2}, I: {3}".format(testDATA, task, counts[0], counts[1]))

        print("Accuracy: {0}".format(round(acc,3)))

    return 


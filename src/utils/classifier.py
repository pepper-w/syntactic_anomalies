# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#

"""
Pytorch Classifier class in the style of scikit-learn
Classifiers include Logistic Regression and MLP
"""

import numpy as np
import copy
import logging

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111, cudaEfficient=False, nepoches=4, maxepoch=200):
        # fix seed
        np.random.seed(seed)
        # torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient
        self.nepoches = nepoches
        self.maxepoch = maxepoch
    
    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split*len(X)):]
            devidx = permutation[0:int(validation_split*len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        if self.cudaEfficient:
            trainX = torch.FloatTensor(trainX).cuda()
            trainy = torch.LongTensor(trainy).cuda()
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        else:
            trainX = torch.FloatTensor(trainX)
            trainy = torch.LongTensor(trainy)
            devX = torch.FloatTensor(devX)
            devy = torch.LongTensor(devy)

        return trainX, trainy, devX, devy

    def fit(self, X, y, validation_data=None, validation_split=None, early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0
        
        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data, validation_split)
        
        # Training
        while not stop_train and self.nepoch<=self.maxepoch:
            self.trainepoch(trainX, trainy, nepoches=self.nepoches)
            accuracy = self.score(devX, devy)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count>=5:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return bestaccuracy
        
    def trainepoch(self, X, y, nepoches=1):
        self.model.train()
        for epoch in range(self.nepoch, self.nepoch + nepoches):
            permutation = np.random.permutation(len(X))
            all_costs   = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.LongTensor(permutation[i:i + self.batch_size])
                # if isinstance(X, torch.cuda.FloatTensor): idx = idx.cuda():
                Xbatch = Variable(X.index_select(0,idx))
                ybatch = Variable(y.index_select(0,idx))
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += nepoches
        
    def score(self, devX, devy):
        self.model.eval()
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor): 
            devX = torch.FloatTensor(devX)
            devy = torch.LongTensor(devy)                
            if self.cudaEfficient:
                devX = devX.cuda()
                devy = devy.cuda()            
        for i in range(0, len(devX), self.batch_size):
            with torch.no_grad():
                Xbatch = Variable(devX[i:i + self.batch_size])
                ybatch = Variable(devy[i:i + self.batch_size])
                # Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
                # ybatch = Variable(devy[i:i + self.batch_size], volatile=True)
            if self.cudaEfficient:
                Xbatch = Xbatch.cuda()
                ybatch = ybatch.cuda()
            output = self.model(Xbatch)
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum()
        accuracy = 1.0*correct / len(devX)
        return accuracy
    
    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX)
        if self.cudaEfficient:
            devX = devX.cuda()
        yhat = np.array([])
        for i in range(0, len(devX), self.batch_size):
            with torch.no_grad():
                Xbatch = Variable(devX[i:i + self.batch_size])
            output = self.model(Xbatch)
            yhat = np.append(yhat, output.data.max(1)[1].squeeze().cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat
    
    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX)
        if self.cudaEfficient:
            devX = devX.cuda()        
        for i in range(0, len(devX), self.batch_size):
            with torch.no_grad(): 
                # Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
                Xbatch = Variable(devX[i:i + self.batch_size])
            if len(probas) == 0:
                probas = self.model(Xbatch).data.cpu().numpy()
            else:
                probas = np.concatenate((probas, self.model(Xbatch).data.cpu().numpy()), axis=0)
        return probas
"""
Logistic Regression with Pytorch
"""
class LogReg(PyTorchClassifier):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg, batch_size, seed, cudaEfficient)
        if cudaEfficient: 
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
                nn.Softmax()
                ).cuda()
            self.loss_fn = nn.CrossEntropyLoss().cuda()
        else: 
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
                nn.Softmax()
                )
            self.loss_fn = nn.CrossEntropyLoss()

        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=self.l2reg)

"""
MLP with Pytorch
"""
class MLP(PyTorchClassifier):
    def __init__(self, inputdim, hiddendim, nclasses, l2reg=0., batch_size=64, seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg, batch_size, seed, cudaEfficient)

        self.hiddendim = hiddendim

        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.hiddendim),
            # TODO : add parameter p for dropout
            nn.Dropout(p=0.25),
            nn.Tanh(),
            nn.Linear(self.hiddendim, self.nclasses),
            nn.Softmax()
            )
        self.loss_fn = nn.CrossEntropyLoss().cuda()
        
        if cudaEfficient:
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()
            
        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=self.l2reg)

        
        

from random import randint
# import matplotlib
import os
import numpy as np
import torch
import nltk
import pickle
import utils.classifier as cl
import encoders.skipthoughts
from encoders.gensen import GenSen, GenSenSingle
import sys
import gc

# import master
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import copy
import random

random.seed(1111)

def embed(model, batch, batch_size, name):
    if (name == 'BOW'):
        with torch.no_grad():
            embeddings = []
            batch = [sent if sent!=[] else ['.'] for sent in batch]
            for sent in batch:
                sentvec = []
                for word in sent:
                    if word in model.word_vec:
                        sentvec.append(model.word_vec[word])
                if not sentvec:
                    sentvec.append(model.word_vec['.'])
                sentvec = np.mean(sentvec, 0)
                embeddings.append(sentvec)
        
            embeddings = np.vstack(embeddings)
            return embeddings
    elif (name == 'InferSent'):
        with torch.no_grad():
            embeddings = model.encode(batch, tokenize=True)
            return embeddings
    elif (name == 'skipthoughts'):
        with torch.no_grad():
            encoder = skipthoughts.Encoder(model)
            embeddings = encoder.encode(batch)
            return embeddings
    elif (name == 'gensen'):
        with torch.no_grad():
            _, embeddings = model.get_representation(
            batch, pool='last', return_numpy=True, tokenize=None)
            return embeddings
    elif (name in {'BERT', 'roberta'}):
        with torch.no_grad():
            (model_bert, tokenizer_bert) = model

            tokenized_batch = tokenizer_bert(text=batch, truncation=True, padding='max_length', return_tensors='pt', max_length=512)
            PAD_id = tokenizer_bert.convert_tokens_to_ids(['[PAD]'])[0] # use attention mask later

            last_hidden_states = model_bert(tokenized_batch.input_ids)[0]  # for CLS pooler: [1]
            output = []

            for i in range(len(tokenized_batch.input_ids)):
              sentence_ids = tokenized_batch.input_ids[i].tolist()
              try: 
                idx_pad = sentence_ids.index(PAD_id)
              except ValueError:
                idx_pad = -1
              output.append(torch.mean(last_hidden_states[i,:idx_pad,:], dim=0))

            embeddings = torch.stack(output, dim=0)

            return embeddings

    elif (name in {'en_fr', 'en_de', 'en_en'}):
        with torch.no_grad():
            (session, embedded_text, text_input) = model
            embeddings = session.run(embedded_text, feed_dict={text_input: batch})
            return embeddings

    else:
        raise NameError('Model not included')
    
def create_embed(model, data, batch_size, name, EMBED_STORE = None):
    print('\nStart embedding for {0}\n'.format(name))
    embed = {'train':{}, 'dev':{}, 'test':{}} 
    for key in embed:
        print('Computing embedding for {0}'.format(key))
        fac = max(1.0, int(len(data[key]['y'])*1.0/(10.0*batch_size)))
        for txt_type in ['X']:
            if (EMBED_STORE is not None):
                fname = './embeddings/embed_' + name +'_' + data +'_' + key + '.txt' 
                if os.path.exists(fname):
                    embed[key][txt_type] = np.loadtxt(fname)
            else:
                embed[key][txt_type] = []
                for ii in range(0, len(data[key]['y']), batch_size):
                    batch = data[key][txt_type][ii:ii + batch_size]
                    embeddings = embed(model, batch, batch_size, name)
                    embed[key][txt_type].append(embeddings)
                    if (ii/batch_size)%(fac) == 0:
                        print("PROGRESS (encoding): {0}%".format(100.0 * ii /len(data[key]['y'])))
                embed[key][txt_type] = np.vstack(embed[key][txt_type])
                if (EMBED_STORE is not None) :
                    np.savetxt(fname, embed[key][txt_type])
                
        print('Computed {0} embeddings\n'.format(key))
    
    return(embed)


class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, X, y, config, outpaths, name):
        self.X = X
        self.y = y
        self.nclasses = config['nclasses']
        self.featdim = self.X['train'].shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier = config['classifier']
        self.nhid = config['nhid']
        self.name = name
        self.outpaths = outpaths
        self.cudaEfficient = False if 'cudaEfficient' not in config else config['cudaEfficient']
        self.modelname = 'sklearn-LogReg' if not config['usepytorch'] else 'pytorch-' + config['classifier']
        self.nepoches = None if 'nepoches' not in config else config['nepoches']
        self.maxepoch = None if 'maxepoch' not in config else config['maxepoch']
        self.noreg = False if 'noreg' not in config else config['noreg']
        
    def run(self):
        print('Training {0}, {1} with standard validation..'.format(self.modelname, self.name))
        regs = [10**t for t in range(-5,-1)] if self.usepytorch else [2**t for t in range(-2,4,1)]
        if self.noreg : regs=[0.]
        scores = []
        for reg in regs:
            if self.usepytorch:
                if self.classifier == 'LogReg':
                    clf = cl.LogReg(inputdim=self.featdim, nclasses=self.nclasses, l2reg=reg, 
		                    seed=self.seed, cudaEfficient=self.cudaEfficient)
                elif self.classifier == 'MLP':
                    clf = cl.MLP(inputdim=self.featdim, hiddendim=self.nhid, nclasses=self.nclasses,                            
		                 l2reg=reg, seed=self.seed, cudaEfficient=self.cudaEfficient)
                if self.nepoches: clf.nepoches = self.nepoches
                if self.maxepoch: clf.maxepoch = self.maxepoch
                clf.fit(self.X['train'], self.y['train'], validation_data=(self.X['valid'], self.y['valid']))
            else:
                clf = LogisticRegression(C=reg, random_state=self.seed)
                clf.fit(self.X['train'], self.y['train'])
            scores.append(np.round(100*clf.score(self.X['valid'], self.y['valid']),2))
        print([('reg:'+str(regs[idx]), scores[idx]) for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        print('Validation : best param found is reg = {0} with score {1}'.format(optreg, devaccuracy))               
        print('Evaluating...')
        if self.usepytorch:
            if self.classifier == 'LogReg':
                clf = cl.LogReg(inputdim = self.featdim, nclasses=self.nclasses, l2reg=optreg,                            
		                seed=self.seed, cudaEfficient=self.cudaEfficient)
            elif self.classifier == 'MLP':
                clf = cl.MLP(inputdim = self.featdim, hiddendim=self.nhid, nclasses=self.nclasses,                          
		             l2reg=optreg, seed=self.seed, cudaEfficient=self.cudaEfficient)
            if self.nepoches: clf.nepoches = self.nepoches
            if self.maxepoch: clf.maxepoch = self.maxepoch
            devacc = clf.fit(self.X['train'], self.y['train'], validation_data=(self.X['valid'], self.y['valid']))
        else:
            # changing solver to multinomial
            clf = LogisticRegression(C=optreg, random_state=self.seed, 
                                     solver = 'sag', multi_class = 'multinomial')
            clf.fit(self.X['train'], self.y['train'])
        
        
        fname = self.outpaths['REGR_MODEL_PATH'] + self.name + self.classifier
        pickle.dump(clf, open(fname, 'wb'))
        clf = pickle.load(open(fname, 'rb'))

        prediction = [x for x in clf.predict(self.X['test']).flatten()]
        confidence = clf.predict_proba(self.X['test']).tolist()
        np.savetxt(self.outpaths['TEST_OUT_PATH'] + 'test_labels_' 
            + self.name + self.classifier, 
            prediction,
            fmt = '%i')
       	np.savetxt(self.outpaths['TEST_OUT_PATH'] + 'test_confs_' 
            + self.name + self.classifier, 
            confidence)
        
        testaccuracy = clf.score(self.X['test'], self.y['test'])
        testaccuracy = np.round(100*testaccuracy, 2)
        return devaccuracy, testaccuracy
    
def trainreg (embed, data, classifier, name, outpaths, useCudaReg):
    # Train
    trainX = embed['train']['X']
    trainY = np.array(data['train']['y'])
    
    # Dev
    devX = embed['dev']['X']
    devY = np.array(data['dev']['y'])
    
    # Test
    testX = embed['test']['X']
    testY = np.array(data['test']['y'])
    
    config_classifier = {'nclasses':2, 'seed':1111, 'usepytorch':True,
                         'classifier': classifier, 'nhid': 512,
                         'cudaEfficient': useCudaReg, 'nepoches':1, 'maxepoch':10}
    
    clf = SplitClassifier(X={'train':trainX, 'valid':devX, 'test':testX},
                          y={'train':trainY, 'valid':devY, 'test':testY},
                          config=config_classifier, outpaths = outpaths,
                          name = name)
    
    devacc, testacc = clf.run()
    print('\nDev acc : {0} Test acc : {1} for classifier\n'.format(devacc, testacc))
    
    return

def trainreg_with_resample (embed, data, classifier, name, outpaths, useCudaReg):
    # Train
    trainX = copy.deepcopy(embed['train']['X'])
    trainY = np.array(data['train']['y'])

    len_train = int(len(trainY) / 2)
    target_index = resample(range(len_train), n_samples=len_train)
    target_index = [2*x for x in target_index]
    target_indeXnother = [x+1 for x in target_index]
    target_index.extend(target_indeXnother)
    # target_index.sort()
    tmp_X = []
    tmp_y = []
    for i in target_index:
        tmp_X.append(trainX[i])
        tmp_y.append(trainY[i])
    trainX = np.array(tmp_X)
    trainY = np.array(tmp_y)

    
    # Dev
    devX = embed['dev']['X']
    devY = np.array(data['dev']['y'])
    
    # Test
    testX = embed['test']['X']
    testY = np.array(data['test']['y'])
    
    config_classifier = {'nclasses':2, 'seed':1111, 'usepytorch':True,
                         'classifier': classifier, 'nhid': 512,
                         'cudaEfficient': useCudaReg}
    
    clf = SplitClassifier(X={'train':trainX, 'valid':devX, 'test':testX},
                          y={'train':trainY, 'valid':devY, 'test':testY},
                          config=config_classifier, outpaths = outpaths,
                          name = name)
    
    devacc, testacc = clf.run()
    print('\nDev acc : {0} Test acc : {1} for classifier\n'.format(devacc, testacc))
    
    return



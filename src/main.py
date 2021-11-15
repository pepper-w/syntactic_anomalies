# coding: utf-8

import sys
import os
from argparse import ArgumentParser
from random import randint
import matplotlib

import numpy as np
import nltk
nltk.download("punkt")
import pickle
import utils.classifier as cl
import trainFuncs as rF
import testFuncs as tF
import utils.loadData as lD

from encoders.infersent import InferSent
import encoders.skipthoughts
from encoders.gensen import GenSen, GenSenSingle
import torch
from transformers import *

import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece


EMBED_STORE = None
batch_size = 64
useCudaReg = True

names = ['en_en', 'en_fr', 'en_de', \
'BOW', 'InferSent', 'skipthoughts', 'gensen', 'BERT', 'roberta']
classifiers = ['LogReg', 'MLP']


def get_label2id(DATA_name):
    if DATA_name == 'SOMO':
        id2label = {0:'O', 1:'C'}
        label2id = {'O': 0, 'C':1}
    elif DATA_name == 'cola':
        id2label = {0:'I', 1:'O'}
        label2id = {'I':0, 'O':1}
    else:
        id2label = {0:'O', 1:'I'}
        label2id = {'O': 0, 'I':1}
    return label2id


def load_model_infersent():
    GLOVE_PATH = 'path/glove.840B.300d.txt' 
    MODEL_PATH = 'path/infersent1.pkl'
    V = 1
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.use_cuda = True
    W2V_PATH = GLOVE_PATH
    model.set_w2v_path(W2V_PATH)
    model.build_vocab_k_words(K=100000)
    return model


def load_model_skipthoughts():
    model = skipthoughts.load_model()
    return model

def load_model_gensen():
    gensen_1 = GenSenSingle(
        model_folder='path/models',
        filename_prefix='nli_large_bothskip',
        pretrained_emb='path/embedding/glove.840B.300d.h5'
    )
    return gensen_1

def load_model_BERT(name):
    if name =='BERT': model_name = 'bert-large-uncased-whole-word-masking'
    if name =='roberta': model_name = 'roberta-large'

    model_bert = AutoModel.from_pretrained(model_name) 

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return (model_bert, tokenizer)


def load_model_use(name):
    if name == 'en_fr': cache_path = "tfhub_cache_path1" # path to set
    if name == 'en_de': cache_path = "tfhub_cache_path2" # path to set
    if name == 'en_en': cache_path = "tfhub_cache_path3" # path to set
    # Set up graph.
    g = tf.Graph()
    with g.as_default():
      text_input = tf.placeholder(dtype=tf.string, shape=[None])
      embedded_text = hub.Module(cache_path)(text_input)
      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    # Initialize session.
    session = tf.Session(graph=g)
    session.run(init_op)

    return (session, embedded_text, text_input)

def get_model(name):
    if name in {'InferSent','BOW'}:
        model = load_model_infersent()
        return model
    if name == 'skipthoughts':
        model = load_model_skipthoughts()
        return model
    if name == 'gensen':
        model = load_model_gensen()
        return model
    if name in {'BERT', 'roberta'}:
        model = load_model_BERT()
        return model
    if name in {'en_fr', 'en_de', 'en_en'}:
        model = load_model_use(name)
        return model


def allClassifiersExist(name, classifiers, REGR_MODEL_PATH):
    flag = True
    # TEST_OUT_PATH
    for classifier in classifiers:
        flag *= os.path.exists(REGR_MODEL_PATH + name + classifier)
    return flag

def train_classifiers(name, model, training_data, REGR_MODEL_PATH, outpaths):

    if (not allClassifiersExist(name, classifiers, REGR_MODEL_PATH)):
        embeddings = rF.create_embed(model, training_data, 
            batch_size, name, EMBED_STORE)
    for classifier in classifiers:
        if (not os.path.exists(REGR_MODEL_PATH + name + classifier)):
            print ('model not exists:{}'.format(REGR_MODEL_PATH + name + classifier))
            rF.trainreg(embeddings, training_data, 
                classifier, name, outpaths, useCudaReg)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("train_tasks", '--names-list', nargs='+', default=[], help="train task dir name")
    parser.add_argument("test_tasks", '--names-list', nargs='+', default=[], help="test task dir name")
    parser.add_argument("data_dir", help="data dir path")

    args = parser.parse_args()

    DATAs = train_tasks
    testDATAs = test_tasks

    tasks = ['dev','test']

    for DATA in DATAs:
        REGR_MODEL_PATH = 'model_path' # path to set
        TEST_OUT_PATH = 'out_path' # path to set
        DATA_PATH = data_dir # path to set
        if not os.path.exists(REGR_MODEL_PATH):
            os.makedirs(REGR_MODEL_PATH)
        if not os.path.exists(TEST_OUT_PATH):
            os.makedirs(TEST_OUT_PATH)
        assert os.path.exists(DATA_PATH), 'data path not found'
        label2id = get_label2id(DATA)
        training_data = lD.loadData(DATA_PATH, DATA, label2id)
        outpaths = {'REGR_MODEL_PATH': REGR_MODEL_PATH, 'TEST_OUT_PATH': TEST_OUT_PATH, 'TEST_DATA_PATH' : DATA_PATH}
        for name in names:
            model = get_model(name)
            train_classifiers(name, model, training_data, REGR_MODEL_PATH, outpaths)
            for testDATA in testDATAs:
                TEST_DATA_PATH = data_dir
                outpaths['TEST_DATA_PATH'] = TEST_DATA_PATH
                print("Running tests for DATA:{} testDATA:{} tasks:{} encoder_name:{} ".format(DATA, testDATA, tasks, name))
                label2id = get_label2id(testDATA)
            for classifier in classifiers:
                tF.runtests(name, classifier, model, tasks, outpaths, label2id, testDATA)


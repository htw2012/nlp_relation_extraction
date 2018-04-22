#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2018/2/13
"""
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import numpy as np

def limit_gpu_usage(use_ratio=0.5):
    '''
    限制GPU使用的比例
    :param use_ratio:
    :return:
    '''
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = use_ratio
    set_session(tf.Session(config=config))


def use_callbacks(training_history_filename="training_history.log", model_name_fmt='trained_models/weibo__weights.{epoch:02d}-{val_loss:.2f}.hdf5', use_lr_strategy=True, save_best=True):
    '''
    使用一些模型的记录策略
    :param training_history_filename:
    :param model_name_fmt:
    :param use_lr_strategy:
    :param save_best:
    :return:
    '''
    # training_history_filename = 'trained_models/weibo/training_history.log'
    callbacks = []
    csv_logger = CSVLogger(training_history_filename, append=False)
    callbacks.append(csv_logger)

    model_names = (model_name_fmt)
    model_checkpoint = ModelCheckpoint(model_names,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=save_best,
                                       save_weights_only=False)
    callbacks.append(model_checkpoint)

    if use_lr_strategy:
        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
        callbacks.append(reduce_learning_rate)
    return callbacks


def create_need_folder(corpus_type, cur_folder="./"):
    import os
    trained_model_folder = cur_folder+"trained_models/%s/"%(corpus_type)
    if os.path.exists(trained_model_folder):
        print("trained_model exists in path: %s"%(trained_model_folder))
    else:
        print("start to create  trained_model path :%s" % (trained_model_folder))
        cmd_str = "mkdir -p %s"%(trained_model_folder)
        # os.mkdir(trained_model, 0755)
        if os.system(cmd_str)!=0:
            print "create trained_model error..."

    trained_hist_folder = cur_folder+"logs/%s/"%(corpus_type)
    if os.path.exists(trained_hist_folder):
        print("trained_hist_file exists in path: %s"%(trained_hist_folder))
    else:
        print("start to create  trained_hist_file path :%s" % (trained_hist_folder))
        cmd_str = "mkdir -p %s" % (trained_hist_folder)
        if os.system(cmd_str) != 0:
            print "create trained_hist_file error..."
    return trained_model_folder, trained_hist_folder


def load_word_embedding(word_file, BASE_DIR="/Users/tongwen/lab/data/glove/glove.6B/", dim=100, max_vocab=5000):
    '''

    :param word_file:
    :param dim:
    :return:
    '''
    import os
    import json

    print('Indexing word vectors.')
    embeddings_index = {} # 所有的WE
    with open(word_file) as fr, open(os.path.join(BASE_DIR, 'glove.6B.%dd.txt'%dim)) as f:
        # step1. 获取所有的词语
        from collections import OrderedDict
        data = json.load(fr, object_pairs_hook=OrderedDict)
        # step2. 获取以后的词汇
        for line in f:
            values = line.split()
            word = values[0]
            if data.get(word):
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
                # print "word", word, "embedding", embeddings_index[word], "type", type(coefs), "shape", coefs.shape

        # step3. 增加UNK词汇到glove
        for k, v in data.items():
            if not embeddings_index.has_key(k):
                # print k, v
                embeddings_index[k] = (np.random.rand(dim)-0.5) * 0.02 # np.random(dim)

        # step4. 按照原始的word顺序，矩阵形式返回
        num_words = min(len(data), max_vocab)
        embedding_matrix = np.zeros((num_words, dim))
        for arg, (k, v) in enumerate(data.items()):
            if arg == num_words:
                break
            if arg == 0:
                print("current print", k, v, "embedding", embeddings_index[k])
            embedding_matrix[arg] = embeddings_index[k]
    print('Found %s word vectors.' % len(embeddings_index), "return-matrix size", len(embedding_matrix))
    return embedding_matrix

def print_best(ret):

    hist = ret.history
    acc_arr = np.array(hist.get('acc'))
    val_acc_arr = np.array(hist.get('val_acc'))
    val_loss_arr = np.array(hist.get('val_loss'))
    loss_arr = np.array(hist.get('loss'))
    f1score_arr = np.array(hist.get('f1_score'))
    epoch_seq = [i + 1 for i in range(len(hist.get('acc')))]
    import pandas as pd
    df = pd.DataFrame({
        "acc": acc_arr,
        "loss": loss_arr,
        "val_acc": val_acc_arr,
        "val_loss": val_loss_arr,
        "f1_score": f1score_arr,
        "epochs": epoch_seq
    })
    # print(df.head(n=10))
    data_sort = df.sort_values(by='f1_score', ascending=False)
    head_data = data_sort.head(n=1)
    print('best_f1score', head_data['f1_score'].as_matrix()[0], "epochs", head_data['epochs'].as_matrix()[0], "val_acc",
          head_data['val_acc'].as_matrix()[0])


def get_class_weight(y_train):
    '''y_train is one hot vector'''
    y_data = np.array([np.argmax(arr) for arr in y_train])
    from sklearn.utils import class_weight
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_data), y_data)
    return class_weight

if __name__ == '__main__':
    # create_need_folder("ag_news_csv2")
    word_file = "/Users/tongwen/PycharmProjects/prepared_work/conv_test/data/dbpedia_csv/word_dict.json"
    load_word_embedding(word_file, dim=100)
    # a = (np.random.rand(10)-0.5) * 0.02
    # print a
    # print a.shape
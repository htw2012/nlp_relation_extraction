#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "SemEval2010的数据的评估"
author= "huangtw"
mtime= 2018/3/10
"""
import json
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class WordItem(object):
    def __init__(self, word, positions=[]):
        self.word = word # 句子分词后
        self.positions = positions

    def __str__(self):
        s_pos = [str(s) for s in self.positions]
        s_str = " ".join(s_pos)
        line = "(word:%s,positions:%s)"%(self.word, s_str)
        return line

class SampleItem(object):

    def __init__(self, word_items, e1, e2, relation_type, direct=True):
        self.word_items = word_items # 句子分词后
        self.e1 = e1
        self.e2 = e2
        self.relation_type = relation_type
        self.direct = direct
    def __str__(self):
        sent = [str(item) for item in self.word_items]
        sent = " ".join(sent)
        line = "sentence-word:%s\ne1:%s, e2:%s, relation_type:%s\ndirect:%s"%(sent, self.e1, self.e2, self.relation_type, str(self.direct))
        return line

def load_src_semevl2010_load_traindata(trainfile):
    '''
    加载原始数据
    :param trainfile:
    :return:
    '''
    with open(trainfile) as fr:
        lines = fr.readlines()
        samples = []

        for i in range(0, len(lines),  4):
            s_line = lines[i].strip()
            entities, words = line_processor(s_line)
            relation_line = lines[i + 1].strip()
            direct = True
            if relation_line != "Other":
                # deal with e1 and e2 with relation_line
                idx_1 = relation_line.find("(")
                idx_2 = relation_line.find(",")
                e1 = relation_line[idx_1+1:idx_2]
                relation_line = relation_line[:idx_1]
                if e1 == "e2":
                    direct = False

            sample = SampleItem(words, e1=entities[0], e2=entities[1], relation_type=relation_line, direct=direct)
            samples.append(sample)
            # if i == 0:
            #     print sample.__str__()
            #     break
        return samples

def load_src_semevl2010_load_testdata(testfile):
    '''
    加载原始数据
    :param testfile:
    :return:
    '''
    with open(testfile) as fr:
        x_datas = []
        for s_line in fr:
            # print(s_line)
            entities, words = line_processor(s_line)
            x_datas.append((words, entities))

        return x_datas

def word_norm(word):
    '''
    词语的简化处理（单词级别）
    :param word:
    :return:
    '''
    word = word.strip()
    word = word.lower()
    word = word.replace(" ", "")
    word = word.replace(",", "")
    word = word.replace("!", "")
    word = word.replace(")", "")
    word = word.replace(".", "")
    word = word.replace("'", "")
    word = word.replace("(", "")
    word = word.replace("*", "")
    word = word.replace(":", "")
    return word

def line_processor(s_line):
    strs = s_line.split("\t")
    text = strs[1]
    text = text.replace("\"", "")
    # get entity for E1 and E2
    fbidx = text.find("<e1>")
    feidx = text.find("</e1>")
    sbidx = text.find("<e2>")
    seidx = text.find("</e2>")
    fpart = text[:fbidx].strip()
    e1part = text[fbidx + len("<e1>"):feidx]
    midpart = text[feidx + len("</e2>"):sbidx].strip()
    e2part = text[sbidx + len("<e1>"):seidx]
    lpart = text[seidx + len("</e2>"):]
    # print("fpart", fpart, "e1part", e1part, "midpart", midpart, "e2part", e2part, "lpart", lpart)
    wordItems = []
    entities = []

    e1_idx = -1
    e2_idx = -1

    idx = -1
    for w in fpart.split(" "):

        w = word_norm(w)
        if w == "": continue
        wordItem = WordItem(w)
        wordItems.append(wordItem)
        idx += 1

    for w in e1part.split(" "):
        w = word_norm(w)
        if w == "": continue
        wordItem = WordItem(w)
        wordItems.append(wordItem)
        entities.append(w)
        idx += 1
        e1_idx = idx

    for w in midpart.split(" "):
        w = word_norm(w)
        if w == "": continue
        wordItem = WordItem(w)
        wordItems.append(wordItem)
        idx += 1


    for w in e2part.split(" "):
        w = word_norm(w)
        if w == "": continue
        wordItem = WordItem(w)
        wordItems.append(wordItem)
        entities.append(w)
        idx += 1
        e2_idx = idx

    for w in lpart.split(" "):
        w = word_norm(w)
        if w == "": continue
        wordItem = WordItem(w)
        wordItems.append(wordItem)
        idx += 1
    # print("e1-indx", e1_idx, "e2-idx", e2_idx)
    for arg, wordItem in enumerate(wordItems):
        l1 = e1_idx-arg
        l2 = e2_idx-arg
        wordItem.positions = [l1, l2]

    return entities, wordItems


def type_dict_stat(y_datas):
    typee_dict = {}
    for data in y_datas:
        if typee_dict.has_key(data):
            typee_dict[data] += 1
        else:
            typee_dict[data] = 1
    for w in sorted(typee_dict, key=typee_dict.get, reverse=True):
        print("stats->type:", w, "freq", typee_dict[w])


def load_word_file(word_dict_file='word_index.json'):
    f = open(word_dict_file)
    data = json.load(f)
    f.close()
    return data


def get_entity_idx(samples, max_len):
    '''return entity position in sentence'''
    # TODO can do it with adding the padding nums
    entity_idxs = []
    for sample in samples:
        idxs = []
        num_pads = max_len- len(sample.word_items)
        for arg, word_item in enumerate(sample.word_items):
            positions = word_item.positions
            if 0 in positions:
                idxs.append(arg+num_pads)
        entity_idxs.append(idxs)
    return entity_idxs

def get_entity_idx_bk(samples, word_dict):
    entity_idxs = []
    for sample in samples:
        idxs = []
        e1 = sample.e1
        e2 = sample.e2
        # TODO using direct, do not yse
        if word_dict.has_key(e1):
            idxs.append(word_dict[e1])
        else:
            idxs.append(word_dict['UNK'])

        if word_dict.has_key(e2):
            idxs.append(word_dict[e2])
        else:
            idxs.append(word_dict['UNK'])

        entity_idxs.append(idxs)
    return entity_idxs


def get_pos_id(x, max_len):
    '''
    编码位置的id TODO 2*max_len+1
    :param x:
    :param max_len:
    :return:
    '''
    if x < -(max_len-1):
        return 0
    if -(max_len-1) <= x <= (max_len-1):
        return x + max_len
    if x > max_len-1:
        return 2*max_len

def sentence2entity_position(samples, max_len):
    '''
    获得实体的位置信息
    :param samples:
    :return:
    '''
    entity1_idxs = []
    entity2_idxs = []
    for sample in samples:
        idxs1 = []
        idxs2 = []

        for arg, word_item in enumerate(sample.word_items):
            positions = word_item.positions
            idx1 = positions[0]
            idx1 = get_pos_id(idx1, max_len)+1
            idxs1.append(idx1)
            
            idx2 = positions[1]
            idx2 = get_pos_id(idx2, max_len)+1
            idxs2.append(idx2)

        entity1_idxs.append(idxs1)
        entity2_idxs.append(idxs2)
    entity1_idxs = np.array(entity1_idxs)
    entity2_idxs = np.array(entity2_idxs)
    return entity1_idxs, entity2_idxs


def load_data(train_file, max_len=200,  word_dict_file="data/sem_eval2010_x.dict", target_dict_file="data/semeval2010_y.dict", mini_freq=2, vocab_size=5000, is_ret_e1e2_idx=False,  is_ret_sentence_pos=False, is_use_direction=False, is_use_tca_cnn=False):
    samples = load_src_semevl2010_load_traindata(train_file)
    # build dict
    if os.path.exists(word_dict_file):
        print("load from exist word file :%s" % (word_dict_file))
        word_dict = load_word_file(word_dict_file)
        print("word_dict-size", len(word_dict))
    else:
        print("start to build  word file :%s" % (word_dict_file))
        word_dict = build_dict_file(word_dict_file, mini_freq, vocab_size, samples)
        print("word_dict-size", len(word_dict))

    # get-max-line
    c_max_len = 0
    for sample in samples:
        l = sample.word_items
        c_max_len = max(len(l), c_max_len)

    print("max_len", c_max_len)
    max_len = max(max_len, c_max_len)
    # step: change word to index
    X_data = []
    for sample in samples:
        # line = line.strip()
        word_items = sample.word_items
        idxs = []
        for word_item in word_items[:max_len]:
            word = word_item.word
            if word_dict.has_key(word):
                idxs.append(word_dict[word])
            else:
                idxs.append(word_dict['UNK'])
        X_data.append(idxs)

    X_data = np.array(X_data, dtype=object)

    if os.path.exists(target_dict_file):
        print("load from exist target_dict_file :%s" % (target_dict_file))
        y_dict = load_word_file(target_dict_file)
    else:
        print("start to build  target_dict_file :%s" % (target_dict_file))
        y_dict = build_y_dict(target_dict_file, samples, is_use_direction=is_use_direction)
    print("ydict_size", len(y_dict))

    # build Y-index
    Y_data = []
    for sample in samples:
        line = sample.relation_type.strip()
        if is_use_direction and not sample.direct:
            Y_data.append(y_dict[line+".r"])
        else:
            Y_data.append(y_dict[line])

    if is_use_tca_cnn:
        sent2e1, sent2e2 = sentence2entity_position(samples, max_len)
        print("sent2e1-shape", sent2e1.shape)
        # Y = [1] * len(Y_data)
        Y = np.array(Y_data).astype('int32')
        Y_data = np.array(Y_data).astype('int32')
        X_data = [X_data, sent2e1, sent2e2, Y_data]
        return (X_data, Y)

    Y_data = np.array(Y_data, dtype=object)
    enc = OneHotEncoder()
    Y_data_arr = enc.fit_transform(Y_data[:, np.newaxis]).toarray()
    if is_ret_e1e2_idx and is_ret_sentence_pos: #
        entity_idx = get_entity_idx(samples, max_len) # samples *2
        entity_idx = np.array(entity_idx)
        print("entity_idx-shape", entity_idx.shape)

        sent2e1, sent2e2 = sentence2entity_position(samples, max_len)
        print("sent2e1-shape", sent2e1.shape)

        X_data = [X_data, sent2e1, sent2e2, entity_idx]
        return (X_data, Y_data_arr)

    if is_ret_e1e2_idx: # 主要用于在WE上扩容E1和E2的维度，使得WE的矩阵变为原始的三倍
        entity_idx = get_entity_idx(samples, max_len) # samples *2
        entity_idx = np.array(entity_idx)
        print("entity_idx-shape", entity_idx.shape)
        X_data = [X_data, entity_idx]
        return (X_data, Y_data_arr)



    if is_ret_sentence_pos:
        sent2e1, sent2e2 = sentence2entity_position(samples, max_len)
        print("sent2e1-shape", sent2e1.shape)

        X_data = [X_data, sent2e1, sent2e2]
        # return X_data

    '''
    from sklearn.model_selection import train_test_split
    x_train, x_dev, y_train, y_dev = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_dev, y_dev)
    '''
    return (X_data, Y_data_arr)

def build_y_dict(target_dict_file, samples, is_use_direction=False):
    y_freq = {}
    for sample in samples:
        y = sample.relation_type
        if is_use_direction and not sample.direct:
            y = sample.relation_type+".r"

        if y_freq.has_key(y):
            y_freq[y] += 1
        else:
            y_freq[y] = 1
    y_dict = {}
    for arg, k in enumerate(sorted(y_freq, key=y_freq.get, reverse=True)):
        y_dict[k] = arg
    # 序列化一次
    with open(target_dict_file, 'w') as fw:
        json.dump(y_dict, fw)
    return y_dict


def load_datas(train_file, test_file, max_len=87, word_dict_file="../data/semeval2010/sem_eval2010_word.dict", target_dict_file="../data/semeval2010/semeval2010_target.dict",
              mini_freq=2, vocab_size=5000, is_ret_e1e2_idx=False, is_ret_sentence_pos=False, is_use_direction=False, is_use_tca_cnn=False):

    (x_train, y_train) = load_data(train_file, word_dict_file=word_dict_file, target_dict_file=target_dict_file, mini_freq=mini_freq, vocab_size=vocab_size, max_len=max_len,is_ret_e1e2_idx=is_ret_e1e2_idx, is_ret_sentence_pos=is_ret_sentence_pos, is_use_direction=is_use_direction, is_use_tca_cnn=is_use_tca_cnn)
    (x_test, y_test) = load_data(test_file, word_dict_file=word_dict_file, target_dict_file=target_dict_file, mini_freq=mini_freq, vocab_size=vocab_size, max_len=max_len,is_ret_e1e2_idx=is_ret_e1e2_idx, is_ret_sentence_pos=is_ret_sentence_pos, is_use_direction=is_use_direction, is_use_tca_cnn=is_use_tca_cnn)

    return (x_train, y_train), (x_test, y_test)


def build_dict_file(dict_file, mini_freq, vocab_size, samples):
    dict_freq = {}
    # cur_max_len = 0
    for sample in samples:
        # words = xline[0]
        word_items = sample.word_items
        for item in word_items:
            x = item.word
            if dict_freq.has_key(x):
                dict_freq[x] += 1
            else:
                dict_freq[x] = 1

    word_dict ={}
    word_dict['PAD'] = 0
    word_dict['UNK'] = 1
    word_dict['GO'] = 2
    for arg, k in enumerate(sorted(dict_freq, key=dict_freq.get, reverse=True)):
        freq = dict_freq[k]
        if freq <= mini_freq or len(word_dict) >= vocab_size:
            break
        word_dict[k] = arg + 3

    # 序列化一次
    with open(dict_file, 'w') as fw:
        json.dump(word_dict, fw)
    return word_dict

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

        # step5. build index-to-vec
        inverted_id_word = {}
        for (k, v) in enumerate(data.items()):
            inverted_id_word[v] = k

        # value sort
        embedding_matrix = np.zeros((num_words, dim)) # embedding排序 从0开始 TODO
        for arg, (k, v) in enumerate(data.items()):
            # print("k", k, "value", v)
            if arg == num_words:
                break

            # if arg == 0:
            #     print("current print", k, v, "embedding", embeddings_index[k])
            embedding_matrix[v] = embeddings_index[k]
    print('Found %s word vectors.' % len(embeddings_index), "return-matrix size", len(embedding_matrix))
    return embedding_matrix

if __name__ == '__main__':
    max_features = 5000
    maxlen = 85  # 400

    # embedding_matrix = load_word_embedding("../data/semeval2010/sem_eval2010_word.dict", BASE_DIR="glove/", dim=embedding_dims, max_vocab=max_features)
    print('Loading data...')
    train_file = "../data/semeval2010/TRAIN_FILE.TXT"
    test_file = "../data/semeval2010/TEST_FILE_FULL.TXT"
    (x_train, y_train), (x_test, y_test) = load_datas(train_file, test_file, max_len=maxlen,
                                                      word_dict_file="../data/semeval2010/sem_eval2010_word.dict",
                                                      target_dict_file="../data/semeval2010/semeval2010_target.dict",
                                                      mini_freq=2, vocab_size=max_features, is_ret_e1e2_idx=False,
                                                      is_ret_sentence_pos=True, is_use_direction=True)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)
    y_data = np.array([np.argmax(arr) for arr in y_train])
    print("y_data", y_data.shape)
    print("class-dist", np.unique(y_data))
    from sklearn.utils import class_weight
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_data), y_data)
    print("class_weight", class_weight.shape)
    print("class_weight", class_weight)
    print("class_weight", class_weight*y_data.shape[0])

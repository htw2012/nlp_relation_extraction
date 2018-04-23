#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2018/3/15
"""

from __future__ import print_function

from keras import Model, Input
from keras.preprocessing import sequence
from keras.layers import Activation, concatenate, merge
from keras.optimizers import Adagrad
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from models_layers_helper import TCAInputEmbeddingAttention, TCAScoringLayer
from model_helper import limit_gpu_usage
from sem_eval2010_loader import load_datas, load_word_embedding
limit_gpu_usage(0.5)

max_features=5000
maxlen=85
batch_size=32
embedding_dims=100
filters=1000
kernel_size=4
hidden_dims=250
epochs=100
num_class=19
pos_embedding_dims=80
use_pretrain=False
relation_dim=80

# 3.1 Input Representation
print("step3.1 Input Representation ...")
input = Input(shape=(maxlen,))
print("input-shape", input._keras_shape)
if use_pretrain:
    embedding_matrix = load_word_embedding("../data/semeval2010/sem_eval2010_word.dict", BASE_DIR="glove/", dim=embedding_dims, max_vocab=max_features)
    we = Embedding(max_features, embedding_dims, input_length=maxlen, weights=[embedding_matrix], trainable=True)(input)
else:
    we = Embedding(max_features, embedding_dims, input_length=maxlen, )(input)
print("after embedding", we._keras_shape)  # ((None, 400), 400, 50) # (None, 400, 50)
e1pos_input = Input(shape=(maxlen,))
max_pos_features = 2*maxlen+2
pos_we = Embedding(max_pos_features, pos_embedding_dims, input_length=maxlen)
entity1_pos_we = pos_we(e1pos_input)
print("entity1_pos_we", entity1_pos_we._keras_shape)
e2pos_input = Input(shape=(maxlen,))
entity2_pos_we = pos_we(e2pos_input)
print("entity2_pos_we", entity2_pos_we._keras_shape)
sentence_we = concatenate([we, entity1_pos_we, entity2_pos_we], axis=-1) # this is matrix S
print("sentence_we", sentence_we._keras_shape) # shape=(?, 85, 460)


# 3.2 Input Attention
print("3.2 Input Attention")
print("step3.2-1. build relation embedding")
relation_input = Input(shape=(1,))
relation_we = Embedding(num_class, relation_dim, input_length=1)
r = relation_we(relation_input)
print("relation_we shape", r._keras_shape)

# step3.2-2. InputEmbeddingAttention get Matrix Q=S*A, need return A
print("step3.2-2. InputEmbeddingAttention")
attention_diag = TCAInputEmbeddingAttention()([sentence_we, r]) # this is matrix attention_diag
print("attention_diag shape", attention_diag._keras_shape)

# step3.2-3. fuse attention
print("step3.2-3. fuse attention")
q_layer = merge([sentence_we, attention_diag], mode='mul') # Q=S*A
print("q_layer shape", q_layer._keras_shape)


# step 3.3 Sentence Representation
print("step 3.3 Sentence Representation")
print("step 3.3.1 Conv1D...")
# Sentence Representation, using the context
c_layer = Conv1D(filters, kernel_size, padding='same', strides=1)(q_layer)
c_layer = Activation(activation='tanh')(c_layer)
print("c_layer", c_layer._keras_shape)

print("step 3.3.2 sentence representation (by GlobalMaxPlooling)")
o_s = GlobalMaxPooling1D()(c_layer) # O_s  sentence representation dim, dim is filters size
print("o_s-shape", o_s._keras_shape) # (?, 1000)

print("step 3.4 Scoring")
# step 3.4. cal score, there's diff, using (oS)⊤Ur #
zeta = TCAScoringLayer()([o_s, r])
print("zeta-shape", zeta._keras_shape)

model = Model(inputs=[input, e1pos_input, e2pos_input, relation_input], outputs=zeta)
adaGrad = Adagrad(lr=0.002) # 初始lr
model.compile(loss='categorical_hinge', optimizer=adaGrad, metrics=['accuracy'])  # TODO loss

model.summary()
# print("type", type(x_train))
print('Loading data...')
train_file = "../data/semeval2010/TRAIN_FILE.TXT"
test_file = "../data/semeval2010/TEST_FILE_FULL.TXT"
(x_train, y_train), (x_test, y_test) = load_datas(train_file, test_file, max_len=maxlen,
                                                  word_dict_file="../data/semeval2010/sem_eval2010_word.dict",
                                                  target_dict_file="../data/semeval2010/semeval2010_target.dict",
                                                  mini_freq=2, vocab_size=max_features, is_use_tca_cnn=True)
print('Pad sequences (samples x time)')
x_data_train = x_train[0]
entity1_idx_train = x_train[1]
entity2_idx_train = x_train[2]
relation_train = x_train[3]

x_data_test = x_test[0]
entity1_idx_test = x_test[1]
entity2_idx_test = x_test[2]
relation_test = x_test[3]

x_train = sequence.pad_sequences(x_data_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_data_test, maxlen=maxlen)

entity1_idx_train = sequence.pad_sequences(entity1_idx_train, maxlen=maxlen)
entity2_idx_train = sequence.pad_sequences(entity2_idx_train, maxlen=maxlen)
entity1_idx_test = sequence.pad_sequences(entity1_idx_test, maxlen=maxlen)
entity2_idx_test = sequence.pad_sequences(entity2_idx_test, maxlen=maxlen)
print('x_data_train shape:', x_train.shape)
print('entity1_idx_train shape:', entity1_idx_train.shape)
print('entity2_idx_train shape:', entity2_idx_train.shape)
print('relation_train shape:', relation_train.shape)
print('y_train shape:', y_train.shape)

print('x_data_test shape:', x_test.shape)
print('entity1_idx_test shape:', entity1_idx_test.shape)
print('entity2_idx_test shape:', entity2_idx_test.shape)
print('relation_test shape:', relation_test.shape)
print('y_test shape:', y_test.shape)

print('Build model...')

ret = model.fit([x_train, entity1_idx_train, entity2_idx_train, relation_train], y_train,
                batch_size=batch_size, epochs=epochs,
                validation_data=([x_test, entity1_idx_test, entity2_idx_test, relation_test], y_test), verbose=1)
print("ret", ret)




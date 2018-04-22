#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2018/3/15
"""

from __future__ import print_function

from keras import Model, Input
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing import sequence
from keras.layers import concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from sem_eval2010_loader import load_datas, load_word_embedding

from models_layers_helper import FirstLevelAttention,  FirstLevelFuse, SecondLevelAttention, SecondAttentionFuse, DeltaDistanceLayer
from model_helper import limit_gpu_usage,print_best
limit_gpu_usage(0.5)


max_features=5000
maxlen=85
batch_size=32
embedding_dims=100
nb_filters=1000
kernel_size=3
hidden_dims=250
epochs=100
num_class=19
concat_pooling=False
pos_embedding_dims=80
attention_type='avg'
use_pretrain=False
relation_dim=80

# step1. build sentence Embbedding
print("step1. build sentence Embbedding")
input = Input(shape=(maxlen,))
print("input-shape", input._keras_shape)
print("sentence-input_tensor-shape", input._keras_shape, "type", type(input))
if use_pretrain:
    embedding_matrix = load_word_embedding("../data/semeval2010/sem_eval2010_word.dict", BASE_DIR="glove/", dim=embedding_dims, max_vocab=max_features)
    sentence_matrix = Embedding(max_features, embedding_dims, input_length=maxlen, weights=[embedding_matrix], trainable=True)(input)
else:
    sentence_matrix = Embedding(max_features, embedding_dims, input_length=maxlen, )(input)
print("after embedding", sentence_matrix._keras_shape)  # ((None, 400), 400, 50) # (None, 400, 50)
# get sentence embedding with position embedding
e1pos_input = Input(shape=(maxlen,))
max_pos_features = 2*maxlen+2
pos_we = Embedding(max_pos_features, pos_embedding_dims, input_length=maxlen)
entity1_pos_we = pos_we(e1pos_input)
print("entity1_pos_we", entity1_pos_we._keras_shape)
e2pos_input = Input(shape=(maxlen,))
entity2_pos_we = pos_we(e2pos_input)
print("entity2_pos_we", entity2_pos_we._keras_shape)
sentence_matrix = concatenate([sentence_matrix, entity1_pos_we, entity2_pos_we], axis=-1) # this is matrix S
print("sentence_matrix", sentence_matrix._keras_shape)

# step2. get first level diag attention (sentence word and entity)
print("step2. get first level diag attention (sentence word and entity)")
position_size = 2
position_input = Input(shape=(position_size,))
print("position_input_tensor-shape", position_input._keras_shape, "type", type(input))
attention_diag = FirstLevelAttention()([sentence_matrix, position_input]) # this is matrix attention_diag
print("attention_diag", attention_diag._keras_shape)

# step3. fuse first-attention matrix and source sentence matrix
print("step3. fuse first-attention matrix and source sentence matrix")
R_layer = FirstLevelFuse(fuse_type='mul')([sentence_matrix, attention_diag])
print("r_layer", R_layer._keras_shape)

# steo4. using conv to recognize the short phrases such as trigrams,
R_star_layer = Conv1D(nb_filters, kernel_size, padding='same', activation='tanh', strides=1)(R_layer)
print("R_star_layer", R_star_layer._keras_shape)

# step5. feed with second attention (attention-based pooling),like bilinear function
print("step5. feed with second attention (attention-based pooling),like bilinear function")
input_rel_full = Input(shape=(num_class,)) #
relation_we = Embedding(num_class, relation_dim, input_length=1) # need to pass the full relation tensor TODO
W_l = relation_we(input_rel_full) # full relation layer 拆分更细一点
Att_p = SecondLevelAttention()([R_star_layer, W_l]) # G=trans(R_star_layer)*U*W_l -> softmax
print("attention-based pooling", Att_p._keras_shape)

# step6. fuse the second attention to highlight the phase-level components
print("step6. fuse the second attention to highlight the phase-level components")
score_tmp = SecondAttentionFuse()([R_star_layer, Att_p])#  the dimension is 1000 *19
print("second fuse attention-based pooling", score_tmp._keras_shape)

# step7. get the output representation,like the formula (12)
print("step7. get the output representation,like the formular (12)")
W_o = GlobalMaxPooling1D()(score_tmp) #
print("output representation", W_o._keras_shape)

# step8. get the novel distance function for classification objective delta公式的计算
print("step8. get the novel distance function for classification objective ")
relation_input = Input(shape=(1,))
r_out = relation_we(relation_input) # 求个映射
print("r_shape", r_out)
novel_distance = DeltaDistanceLayer()([W_o, r_out]) # TODO the Dimensions in not match , [?,1,80], [?,19]

model = Model(inputs=[input, e1pos_input, e2pos_input, position_input, relation_input], outputs=novel_distance)
model.compile(loss='categorical_hinge', optimizer='adam', metrics=['accuracy'])  # TODO
model.summary()

print('Loading data...')
train_file = "../data/semeval2010/TRAIN_FILE.TXT"
test_file = "../data/semeval2010/TEST_FILE_FULL.TXT"
(x_train, y_train), (x_test, y_test) = load_datas(train_file, test_file, max_len=maxlen,
                                                  word_dict_file="../data/semeval2010/sem_eval2010_word.dict",
                                                  target_dict_file="../data/semeval2010/semeval2010_target.dict",
                                                  mini_freq=2, vocab_size=max_features, is_ret_e1e2_idx=True,
                                                  is_ret_sentence_pos=True, is_use_direction=True)
print('Pad sequences (samples x time)')
x_data_train = x_train[0]
entity1_idx_train = x_train[1]
entity2_idx_train = x_train[2]
position_train = x_train[3]

x_data_test = x_test[0]
entity1_idx_test = x_test[1]
entity2_idx_test = x_test[2]
postion_test = x_test[3]

x_train = sequence.pad_sequences(x_data_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_data_test, maxlen=maxlen)

entity1_idx_train = sequence.pad_sequences(entity1_idx_train, maxlen=maxlen)
entity2_idx_train = sequence.pad_sequences(entity2_idx_train, maxlen=maxlen)
entity1_idx_test = sequence.pad_sequences(entity1_idx_test, maxlen=maxlen)
entity2_idx_test = sequence.pad_sequences(entity2_idx_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_train shape:', x_train.shape)
print('entity1_idx_train shape:', entity1_idx_train.shape)
print('entity2_idx_train shape:', entity2_idx_train.shape)

print('Build model...')

callbacks = []
training_history_filename = "results/attention_two_level_training_history_pos_we%d_filters%d.csv"%(embedding_dims, nb_filters)
csv_logger = CSVLogger(training_history_filename, append=False)
callbacks.append(csv_logger)
model_name_fmt = "trained_models/attention_two_level_we_%d_filters%d.{epoch:02d}-{val_acc:.4f}.hdf5"%(embedding_dims, nb_filters)
model_names = (model_name_fmt)
model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)
callbacks.append(model_checkpoint)

from model_metrics_helper import F1ScoreOfficial
f1score = F1ScoreOfficial(([x_test, entity1_idx_test, entity2_idx_test, postion_test], y_test))
callbacks.append(f1score)


ret = model.fit([x_train, entity1_idx_train, entity2_idx_train, position_train], y_train,
                batch_size=batch_size, epochs=epochs,
                validation_data=([x_test, entity1_idx_test, entity2_idx_test, postion_test], y_test), verbose=2, callbacks=callbacks)
import pandas as pd
data = pd.read_csv(training_history_filename)

describe = data.describe()
best_acc = describe.iloc[7, [3]]
print("best_acc", best_acc)

print_best(ret)
# max_ix = hist['val_acc'].index(best_acc)
# print("epoch:%d,best-val_acc:%.4f"%(max_ix, hist['val_acc'][max_ix]))



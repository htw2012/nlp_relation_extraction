#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2018/3/16
"""

from keras.layers import concatenate
from keras import layers
from keras import backend as K
from keras import initializers

class InfoEnlargeEmbedding(layers.Layer):
    # 50如何得到
    def compute_output_shape(self, input_shape):
        print("type-input_shape", type(input_shape))
        index_shape = input_shape[1]
        input_shape = input_shape[0]
        print("input_shape2", index_shape)
        ret = (None, input_shape[1], input_shape[2] * (1 + index_shape[1]))
        print("compute_output_shape-after", ret)
        return ret

    def call(self, inputs):

        print("x-0", inputs[0], "x-1", inputs[1])
        idxs = inputs[1]
        x = inputs[0]
        # using slice the value with two tensor what (?, 2, 50)
        if K.dtype(idxs) != 'int32':  # need convert
            idxs = K.cast(idxs, 'int32')
        # z = K.gather(x, idxs) (?, 400, 2, 50) not the right (?, 2, 50)
        max_len = x.shape[1]  # using the max_len

        z = K.one_hot(idxs, max_len)
        # z10 = z*x # [?,2,87], [?,87,50] can't do it !!!
        # z10 = K.prod([z,x]) # 内积，怎么用的
        z10 = K.batch_dot(z,x)
        print("z10", z10)
        new_emd = x.shape[2] * idxs.shape[1]
        z11 = K.reshape(z10, (-1, new_emd))
        print("z11", z11)
        z12 = K.repeat(z11, max_len)
        print("z12", z12)
        z13 = K.concatenate([x,z12], axis=2)
        print("z13", z13)
        return z13

class EmbeddingAttention(layers.Layer):

    def __init__(self, fuzzy_type='concat', **kwargs):
        self.fuzzy_type = fuzzy_type
        super(EmbeddingAttention, self).__init__(**kwargs)

    '''融合方式使用avg的形式'''
    # 输入层的attention
    def compute_output_shape(self, input_shape):

        ret = input_shape[0] #(None, input_shape[1], input_shape[2] * (1 + index_shape[1]))
        print("compute_output_shape-after", ret)
        return ret

    def call(self, inputs):
        print("x-0", inputs[0], "x-1", inputs[1])
        idxs = inputs[1]
        sent_we = inputs[0]
        # using slice the value with two tensor what (?, 2, 50)
        if K.dtype(idxs) != 'int32':  # need convert
            idxs = K.cast(idxs, 'int32')
        # z = K.gather(x, idxs) (?, 400, 2, 50) not the right (?, 2, 50)
        max_len = sent_we.shape[1]  # using the max_len
        we_dim = sent_we.shape[2]

        z = K.one_hot(idxs, max_len)
        # z10 = z*x # [?,2,87], [?,87,50] can't do it !!!
        # z10 = K.prod([z,x]) # 内积，怎么用的
        entity_we = K.batch_dot(z, sent_we)
        print("z10", entity_we) # shape=(?, 2, 50) # entiy WE
        # t_sent_we = K.transpose(sent_we)
        # print("transpose sent_we", t_sent_we)  # shape=(50, 87, ?) # not right
        t_sent_we = K.permute_dimensions(sent_we, (0,2,1))
        att = K.batch_dot(entity_we, t_sent_we)
        print("attention", att)

        # using avg as attention ways
        if self.fuzzy_type == 'avg': # (?, ?)
            att_m = K.mean(att, axis=1)
            print("att_m", att_m) # (?, 87)
        elif self.fuzzy_type == 'mul': # 维度如何降低
            att_m = att[:,1,:] * att[:,0,:]
            print("att_m", att_m)  # (?, 87)
        elif self.fuzzy_type == 'concat':
            att_m1 = K.batch_flatten(att)
            print("att_m1", att_m1)
            raise ValueError('Has not implement the way:%s' %(self.fuzzy_type))
        elif self.fuzzy_type == 'Variant-2':
            att_m1 = att[:,1,:]-att[:,0,:]
            print("att_m1", att_m1)  # (?, 87)
            # att_m2 = K.squeeze(att_m1, axis=1)
            # print("att_m2", att_m2)  # (?, 87)
            att_m = 0.5*att_m1
            print("att_m", att_m)  # (?, 87)
        else:
            raise ValueError('Do not support the way:%s' %(self.fuzzy_type))

        atts = K.repeat(att_m, we_dim) # (?, max_len, 50)
        print("atts", atts)  # (?, 1, 50)
        atts_t = K.permute_dimensions(atts, (0,2,1))
        print("atts-t", atts_t)  # (?, 1, 50)
        print("sent_we", sent_we)
        new_we = sent_we*atts_t # element-wise multi
        # new_we = K.batch_dot(sent_we, atts_t) # check it
        print("new_we", new_we)

        return new_we


class EmbeddingContextAttention(layers.Layer):
    '''
    带context形式的attention形式.
    比如输入max_len*d,经过两个实体的attention矩阵变为
    max_len*kd或者max_len*2kd
    '''

    def __init__(self, fuzzy_type='concat', context_size=3, is_use_norm=False, is_use_scale=False, **kwargs):
        self.fuzzy_type = fuzzy_type
        self.context_size = context_size
        self.is_use_norm = is_use_norm
        self.is_use_scale = is_use_scale

        super(EmbeddingContextAttention, self).__init__(**kwargs)

    '''融合方式使用avg的形式'''
    # 输入层的attention
    def compute_output_shape(self, input_shape):
        sent_len = input_shape[0][1]
        output_dim = self.context_size * input_shape[0][2]
        if self.fuzzy_type == 'concat':
            output_dim = output_dim*2

        ret = (None, sent_len, output_dim)
        print("compute_output_shape-after", ret) # n*k
        return ret

    def call(self, inputs):
        print("x-0", inputs[0], "x-1", inputs[1])
        idxs = inputs[1]
        sent_we = inputs[0]
        # using slice the value with two tensor what (?, 2, 50)
        if K.dtype(idxs) != 'int32':  # need convert
            idxs = K.cast(idxs, 'int32')
        max_len = sent_we.shape[1]  # using the max_len
        we_dim = sent_we.shape[2]

        z = K.one_hot(idxs, max_len)
        # z10 = z*x # [?,2,87], [?,87,50] can't do it !!!
        # z10 = K.prod([z,x]) # 内积，怎么用的
        entity_we = K.batch_dot(z, sent_we)
        print("z10", entity_we) # shape=(?, 2, 50) # entiy WE
        # t_sent_we = K.transpose(sent_we)
        # print("transpose sent_we", t_sent_we)  # shape=(50, 87, ?) # not right
        t_sent_we = K.permute_dimensions(sent_we, (0,2,1))
        att = K.batch_dot(entity_we, t_sent_we)
        print("attention", att)

        # using avg as attention ways
        if self.fuzzy_type == 'avg': # (?, ?)
            att_m = K.mean(att, axis=1)
            print("att_m", att_m) # (?, 87)
        elif self.fuzzy_type == 'mul': # 维度如何降低
            att_m = att[:,1,:] * att[:,0,:]
            print("att_m", att_m)  # (?, 87)
        elif self.fuzzy_type == 'concat':
            att_m = concatenate([att[:,0,:], att[:,1,:]])
            print("att_m_attention", att_m)
        elif self.fuzzy_type == 'Variant-2':
            att_m1 = att[:,1,:]-att[:,0,:]
            print("att_m1", att_m1)  # (?, 87)
            # att_m2 = K.squeeze(att_m1, axis=1)
            # print("att_m2", att_m2)  # (?, 87)
            att_m = 0.5*att_m1
            print("att_m", att_m)  # (?, 87)
        else:
            raise ValueError('Do not support the way:%s' %(self.fuzzy_type))

        # norm
        if self.is_use_norm:
            att_m = K.softmax(att_m) # for the last dim

        if self.fuzzy_type =='concat':
            att1 = K.repeat(att[:,0,:], we_dim*self.context_size)
            att2 = K.repeat(att[:,1,:], we_dim*self.context_size)
            print("att1", att1)
            atts = K.concatenate([att1, att2], axis=-1)
            print("atts", atts)
        else:
            atts = K.repeat(att_m, we_dim*self.context_size) # (?, max_len, 50)

        print("atts", atts)  # (?, 1, 50)
        atts_t = K.permute_dimensions(atts, (0,2,1))
        print("atts-t", atts_t)  # (?, 1, 50)
        print("sent_we", sent_we)

        # change sent_we to context sent TODO
        # 改变计算过程
        context_wes = []
        for i in range(max_len):
            context_we = []
            semi_wind_size = int(self.context_size/2)
            for j in reversed(range(semi_wind_size)):
                # print("mid_we", mid_we)
                first_we = sent_we[:, i-j,:] if i-j>=0 else K.zeros_like(mid_we)
                context_we.append(first_we)
            mid_we = sent_we[:, i, :]
            context_we.append(mid_we)

            for j in range(semi_wind_size):
                last_we = sent_we[:, i + j, :] if i + j < max_len else K.zeros_like(mid_we)
                context_we.append(last_we)

            context_we = concatenate(context_we, axis=-1) # shape=(?, 150)
            # print("context_we", context_we)
            context_we_exp = K.expand_dims(context_we,axis=1)
            # print("context_we_exp", context_we_exp)
            context_wes.append(context_we_exp)
        context_wes = concatenate(context_wes, axis=1) # shape=(?, max_len, 150)
        print("context_wes", context_wes) # (?, 87, 150)
        if self.fuzzy_type!='concat':
            new_we = context_wes * atts_t  # element-wise multi
        else:
            new_we1 = context_wes * atts_t[:,:max_len,:]
            new_we2 = context_wes * atts_t[:,max_len:,:]
            print("new_we1", new_we1)
            new_we = concatenate([new_we1, new_we2], axis=-1)
        # new_we = K.batch_dot(sent_we, atts_t) # check it
        print("new_we", new_we)

        # using scale
        if self.is_use_scale:
            new_dim = we_dim * self.context_size
            # print("new_dim", new_dim, "type", type(new_dim))
            new_dim = new_dim.value
            print("new_dim", new_dim, "type", type(new_dim))

            '''
            # the first way
            tmp_one = K.ones_like(new_we)
            tmp_one *= new_dim
            tmp_one = K.sqrt(tmp_one)
            new_we = new_we * tmp_one
            '''
            import math
            sqrt_val = math.sqrt(new_dim)
            print("new_dim", new_dim, "sqrt_val", sqrt_val)
            new_we = sqrt_val*new_we

        return new_we


class EmbeddingContextAttention(layers.Layer):
    '''
    带context形式的attention形式.
    比如输入max_len*d,经过两个实体的attention矩阵变为
    max_len*kd或者max_len*2kd
    '''

    def __init__(self, fuzzy_type='concat', context_size=3, is_use_norm=False, is_use_scale=False, **kwargs):
        self.fuzzy_type = fuzzy_type
        self.context_size = context_size
        self.is_use_norm = is_use_norm
        self.is_use_scale = is_use_scale

        super(EmbeddingContextAttention, self).__init__(**kwargs)

    '''融合方式使用avg的形式'''
    # 输入层的attention
    def compute_output_shape(self, input_shape):
        sent_len = input_shape[0][1]
        output_dim = self.context_size * input_shape[0][2]
        if self.fuzzy_type == 'concat':
            output_dim = output_dim*2

        ret = (None, sent_len, output_dim)
        print("compute_output_shape-after", ret) # n*k
        return ret

    def call(self, inputs):
        print("x-0", inputs[0], "x-1", inputs[1])
        idxs = inputs[1]
        sent_we = inputs[0]
        # using slice the value with two tensor what (?, 2, 50)
        if K.dtype(idxs) != 'int32':  # need convert
            idxs = K.cast(idxs, 'int32')
        max_len = sent_we.shape[1]  # using the max_len
        we_dim = sent_we.shape[2]

        z = K.one_hot(idxs, max_len)
        # z10 = z*x # [?,2,87], [?,87,50] can't do it !!!
        # z10 = K.prod([z,x]) # 内积，怎么用的
        entity_we = K.batch_dot(z, sent_we)
        print("z10", entity_we) # shape=(?, 2, 50) # entiy WE
        # t_sent_we = K.transpose(sent_we)
        # print("transpose sent_we", t_sent_we)  # shape=(50, 87, ?) # not right
        t_sent_we = K.permute_dimensions(sent_we, (0,2,1))
        att = K.batch_dot(entity_we, t_sent_we)
        print("attention", att)

        # using avg as attention ways
        if self.fuzzy_type == 'avg': # (?, ?)
            att_m = K.mean(att, axis=1)
            print("att_m", att_m) # (?, 87)
        elif self.fuzzy_type == 'mul': # 维度如何降低
            att_m = att[:,1,:] * att[:,0,:]
            print("att_m", att_m)  # (?, 87)
        elif self.fuzzy_type == 'concat':
            att_m = concatenate([att[:,0,:], att[:,1,:]])
            print("att_m_attention", att_m)
        elif self.fuzzy_type == 'Variant-2':
            att_m1 = att[:,1,:]-att[:,0,:]
            print("att_m1", att_m1)  # (?, 87)
            # att_m2 = K.squeeze(att_m1, axis=1)
            # print("att_m2", att_m2)  # (?, 87)
            att_m = 0.5*att_m1
            print("att_m", att_m)  # (?, 87)
        else:
            raise ValueError('Do not support the way:%s' %(self.fuzzy_type))

        # norm
        if self.is_use_norm:
            att_m = K.softmax(att_m) # for the last dim

        if self.fuzzy_type =='concat':
            att1 = K.repeat(att[:,0,:], we_dim*self.context_size)
            att2 = K.repeat(att[:,1,:], we_dim*self.context_size)
            print("att1", att1)
            atts = K.concatenate([att1, att2], axis=-1)
            print("atts", atts)
        else:
            atts = K.repeat(att_m, we_dim*self.context_size) # (?, max_len, 50)

        print("atts", atts)  # (?, 1, 50)
        atts_t = K.permute_dimensions(atts, (0,2,1))
        print("atts-t", atts_t)  # (?, 1, 50)
        print("sent_we", sent_we)

        # change sent_we to context sent TODO
        # 改变计算过程
        context_wes = []
        for i in range(max_len):
            context_we = []
            semi_wind_size = int(self.context_size/2)
            for j in reversed(range(semi_wind_size)):
                # print("mid_we", mid_we)
                first_we = sent_we[:, i-j,:] if i-j>=0 else K.zeros_like(mid_we)
                context_we.append(first_we)
            mid_we = sent_we[:, i, :]
            context_we.append(mid_we)

            for j in range(semi_wind_size):
                last_we = sent_we[:, i + j, :] if i + j < max_len else K.zeros_like(mid_we)
                context_we.append(last_we)

            context_we = concatenate(context_we, axis=-1) # shape=(?, 150)
            # print("context_we", context_we)
            context_we_exp = K.expand_dims(context_we,axis=1)
            # print("context_we_exp", context_we_exp)
            context_wes.append(context_we_exp)
        context_wes = concatenate(context_wes, axis=1) # shape=(?, max_len, 150)
        print("context_wes", context_wes) # (?, 87, 150)
        if self.fuzzy_type!='concat':
            new_we = context_wes * atts_t  # element-wise multi
        else:
            new_we1 = context_wes * atts_t[:,:max_len,:]
            new_we2 = context_wes * atts_t[:,max_len:,:]
            print("new_we1", new_we1)
            new_we = concatenate([new_we1, new_we2], axis=-1)
        # new_we = K.batch_dot(sent_we, atts_t) # check it
        print("new_we", new_we)

        # using scale
        if self.is_use_scale:
            new_dim = we_dim * self.context_size
            # print("new_dim", new_dim, "type", type(new_dim))
            new_dim = new_dim.value
            print("new_dim", new_dim, "type", type(new_dim))

            '''
            # the first way
            tmp_one = K.ones_like(new_we)
            tmp_one *= new_dim
            tmp_one = K.sqrt(tmp_one)
            new_we = new_we * tmp_one
            '''
            import math
            sqrt_val = math.sqrt(new_dim)
            print("new_dim", new_dim, "sqrt_val", sqrt_val)
            new_we = sqrt_val*new_we

        return new_we


class EmbeddingFreeContextAttention(layers.Layer):
    '''
    带context形式的attention形式.
    比如输入max_len*d,经过两个实体的attention矩阵变为
    max_len*kd或者max_len*2kd
    '''

    def __init__(self, fuzzy_type='concat', context_size=1, is_use_norm=False, is_use_scale=False, **kwargs):
        self.fuzzy_type = fuzzy_type
        self.context_size = context_size
        self.is_use_norm = is_use_norm
        self.is_use_scale = is_use_scale

        super(EmbeddingFreeContextAttention, self).__init__(**kwargs)

    '''融合方式使用avg的形式'''
    # 输入层的attention
    def compute_output_shape(self, input_shape):
        sent_len = input_shape[0][1]
        output_dim = self.context_size * input_shape[0][2]
        if self.fuzzy_type == 'concat':
            output_dim = output_dim*2

        ret = (None, sent_len, output_dim)
        print("compute_output_shape-after", ret) # n*k
        return ret

    def call(self, inputs):
        print("x-0", inputs[0], "x-1", inputs[1])
        idxs = inputs[1]
        sent_we = inputs[0]
        # using slice the value with two tensor what (?, 2, 50)
        if K.dtype(idxs) != 'int32':  # need convert
            idxs = K.cast(idxs, 'int32')
        max_len = sent_we.shape[1]  # using the max_len
        we_dim = sent_we.shape[2]

        z = K.one_hot(idxs, max_len)
        # z10 = z*x # [?,2,87], [?,87,50] can't do it !!!
        # z10 = K.prod([z,x]) # 内积，怎么用的
        entity_we = K.batch_dot(z, sent_we)
        print("z10", entity_we) # shape=(?, 2, 50) # entiy WE
        # t_sent_we = K.transpose(sent_we)
        # print("transpose sent_we", t_sent_we)  # shape=(50, 87, ?) # not right
        t_sent_we = K.permute_dimensions(sent_we, (0,2,1))
        att = K.batch_dot(entity_we, t_sent_we)
        print("attention", att)

        # using avg as attention ways
        if self.fuzzy_type == 'avg': # (?, ?)
            att_m = K.mean(att, axis=1)
            print("att_m", att_m) # (?, 87)
        elif self.fuzzy_type == 'mul': # 维度如何降低
            att_m = att[:,1,:] * att[:,0,:]
            print("att_m", att_m)  # (?, 87)
        elif self.fuzzy_type == 'concat':
            att_m = concatenate([att[:,0,:], att[:,1,:]])
            print("att_m_attention", att_m)
        elif self.fuzzy_type == 'Variant-2':
            att_m1 = att[:,1,:]-att[:,0,:]
            print("att_m1", att_m1)  # (?, 87)
            # att_m2 = K.squeeze(att_m1, axis=1)
            # print("att_m2", att_m2)  # (?, 87)
            att_m = 0.5*att_m1
            print("att_m", att_m)  # (?, 87)
        else:
            raise ValueError('Do not support the way:%s' %(self.fuzzy_type))

        # norm
        if self.is_use_norm:
            att_m = K.softmax(att_m) # for the last dim

        if self.fuzzy_type =='concat':
            att1 = K.repeat(att[:,0,:], we_dim*self.context_size)
            att2 = K.repeat(att[:,1,:], we_dim*self.context_size)
            print("att1", att1)
            atts = K.concatenate([att1, att2], axis=-1)
            print("atts", atts)
        else:
            atts = K.repeat(att_m, we_dim*self.context_size) # (?, max_len, 50)

        print("atts", atts)  # (?, 1, 50)
        atts_t = K.permute_dimensions(atts, (0,2,1))
        print("atts-t", atts_t)  # (?, 1, 50)
        print("sent_we", sent_we)

        # change sent_we to context sent TODO
        # 改变计算过程
        context_wes = sent_we
        print("context_wes", context_wes) # (?, 87, 150)
        if self.fuzzy_type!='concat':
            new_we = context_wes * atts_t  # element-wise multi
        else:
            new_we1 = context_wes * atts_t[:,:max_len,:]
            new_we2 = context_wes * atts_t[:,max_len:,:]
            print("new_we1", new_we1)
            new_we = concatenate([new_we1, new_we2], axis=-1)
        # new_we = K.batch_dot(sent_we, atts_t) # check it
        print("new_we", new_we)

        # using scale
        if self.is_use_scale:
            new_dim = we_dim * self.context_size
            # print("new_dim", new_dim, "type", type(new_dim))
            new_dim = new_dim.value
            print("new_dim", new_dim, "type", type(new_dim))

            '''
            # the first way
            tmp_one = K.ones_like(new_we)
            tmp_one *= new_dim
            tmp_one = K.sqrt(tmp_one)
            new_we = new_we * tmp_one
            '''
            import math
            sqrt_val = math.sqrt(new_dim)
            print("new_dim", new_dim, "sqrt_val", sqrt_val)
            new_we = sqrt_val*new_we

        return new_we

class AttLayer(layers.Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        # https://stackoverflow.com/questions/35163789/theano-dimshuffle-equivalent-in-googles-tensorflow
        # dim shufflle http://blog.csdn.net/niuwei22007/article/details/48949869
        eij = K.tanh(K.dot(x, self.W))
        print("eij", eij)
        ai = K.exp(eij)
        print("ai", ai)

        # weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x') # expand_dims(input, 1)
        weights = ai/K.sum(ai, axis=1)
        weights = K.expand_dims(weights, axis=1)

        # weighted_input = x*weights.dimshuffle(0,1,'x') # dim exp?

        weighted_input = x*weights
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

class AttentionLayer(layers.Layer):
    '''单层的Attention'''
    def __init__(self, return_attention=False, **kwargs):
        self.return_attention=return_attention
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("build-input_shape", input_shape)
        # Create a trainable weight variable for this layer.
        self.attention_probs = self.add_weight(name='attention_probs', shape=(input_shape[2], input_shape[2]), initializer='uniform',
                                      trainable=True)
        print("build-input_shape", input_shape)
        super(AttentionLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        print("inputs-shape", inputs.shape, "attention_probs-shape", self.attention_probs.shape)
        eij = K.dot(inputs, self.attention_probs)
        # eij = K.tanh(eij)
        a = K.exp(eij)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # input + attention
        print("inputs-shape", inputs.shape, "a-shape", a.shape)
        weighted_input = inputs * a
        # weighted_input = inputs * K.expand_dims(a)
        # result = K.sum(weighted_input, axis=1)
        print("call-shape", weighted_input.shape)
        if self.return_attention:
            return [weighted_input, a]
        return weighted_input

    def compute_output_shape(self, input_shape):
        print("compute_output_shape-input_shape", input_shape)
        return input_shape
        # return (input_shape[2], input_shape[2])


class AttentionTransLayer(layers.Layer):
    '''TODO 这种Attention改变了维度''' # cross attention
    def __init__(self, return_attention=False, **kwargs):
        self.return_attention=return_attention
        super(AttentionTransLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("build-input_shape", input_shape)
        # Create a trainable weight variable for this layer.
        self.attention_probs = self.add_weight(name='attention_probs', shape=(input_shape[1], input_shape[1]), initializer='uniform',
                                      trainable=True)
        print("build-input_shape", input_shape)
        super(AttentionTransLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        print("inputs-shape", inputs.shape, "attention_probs-shape", self.attention_probs.shape)
        inputs_t = K.permute_dimensions(inputs, (0,2,1))
        eij = K.dot(inputs_t, self.attention_probs)
        # eij = K.tanh(eij)
        a = K.exp(eij)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # input + attention
        print("inputs-shape", inputs.shape, "a-shape", a.shape)
        weighted_input = inputs_t * a
        # weighted_input = inputs * K.expand_dims(a)
        # result = K.sum(weighted_input, axis=1)
        print("call-shape", weighted_input.shape)
        if self.return_attention:
            return [weighted_input, a]
        return weighted_input

    def compute_output_shape(self, input_shape):
        print("compute_output_shape-input_shape", input_shape)
        return input_shape
        # return (input_shape[2], input_shape[2])


# def attention(input,filters):
# 	x=GlobalMaxPooling1D()(input)
# 	x=Dense(64,activation='relu')(x)
# 	attention=Dense(filters,activation='softmax')(x)
# 	attMatrix=RepeatVector(800)(attention)
# 	attention_mul = merge([input, attMatrix], output_shape=[800,filters], mode='mul') #add the attention to the featuremap
# 	return attention_mul

class AttentionChannelWiseLayer(layers.Layer):

    def __init__(self, return_attention=False, attention_dim=64, **kwargs):
        self.return_attention=return_attention
        self.attention_dim = attention_dim
        super(AttentionChannelWiseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("build-input_shape", input_shape)
        # Create a trainable weight variable for this layer.
        self.attention_probs = self.add_weight(name='attention_probs', shape=(input_shape[1], input_shape[1]), initializer='uniform',
                                      trainable=True)
        print("build-input_shape", input_shape)
        super(AttentionChannelWiseLayer, self).build(input_shape)  # Be sure to call this somewhere!



    def call(self, inputs):
        '''
        using channel-wise attention TODO
        :param inputs:
        :return:
        '''
        print("inputs-shape", inputs.shape)
        x = K.max(inputs, axis=1)
        x = K.dot(x, self.attention_probs)
        x = K.relu(x)


        inputs_t = K.permute_dimensions(inputs, (0,2,1))
        eij = K.dot(inputs_t, self.attention_probs)
        # eij = K.tanh(eij)
        a = K.exp(eij)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # input + attention
        print("inputs-shape", inputs.shape, "a-shape", a.shape)
        weighted_input = inputs_t * a
        # weighted_input = inputs * K.expand_dims(a)
        # result = K.sum(weighted_input, axis=1)
        print("call-shape", weighted_input.shape)
        if self.return_attention:
            return [weighted_input, a]
        return weighted_input

    def compute_output_shape(self, input_shape):
        print("compute_output_shape-input_shape", input_shape)
        return input_shape


class TCAInputEmbeddingAttention(layers.Layer):
    '''
    双线性的方式进行输入端的attention，计算方式为trans(w_i)*M*r 得到一个值的向量。
    但是矩阵乘以右对角矩阵，相当于原始矩阵每行元素乘以向量的对应元素
    '''
    def __init__(self,  **kwargs):
        super(TCAInputEmbeddingAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        print("build-input_shape", input_shape)
        sent_we_shape = input_shape[0]
        relation_we_shape = input_shape[1]
        print("build-input_shape, sent_we_shape {}, relation_we_shape {}".format(sent_we_shape, relation_we_shape))
        # Create a trainable weight variable for this layer.
        m_x_dim = sent_we_shape[-1] # 260
        m_y_dim = relation_we_shape[-1] # 80
        print("m_x_dim {}, m_y_dim {}".format(m_x_dim, m_y_dim))
        self.attention_probs = self.add_weight(name='attention_probs', shape=(m_x_dim, m_y_dim), initializer='uniform', trainable=True)
        print("attention_probs", self.attention_probs)
        super(TCAInputEmbeddingAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        '''the detail calulate trans(w_i)*M*r'''
        sent_we = inputs[0] # (None, 85, 460)
        relation_we = inputs[1] # (None, 1, 80)
        print("sent_we {}, relation_we {}".format(sent_we, relation_we))
        # tmp = sent_we*self.attention_probs # (None, 85, 460) * (460, 80)-> # (None, 85, 80)
        # tmp = K.batch_dot(sent_we, self.attention_probs)
        tmp = K.dot(sent_we, self.attention_probs)
        print("tmp_value {}".format(tmp))

        # blow line maybe is error
        s_relation_we = K.squeeze(relation_we, axis=1)
        print("s_relation_we {}".format(s_relation_we))

        eta = K.dot(tmp, s_relation_we)
        eta = K.max(eta, axis=-1)
        # t_relation_we = K.transpose(relation_we) # shape=(80, 1, ?)
        # print("t_relation_we {}".format(t_relation_we))
        # eta = tmp * t_relation_we  # (None, 85, 80) * (1, 80)T
        print("eta {}".format(eta))
        eta = K.exp(eta)
        alpha = eta/K.cast(K.sum(eta, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        attention = K.zeros_like(sent_we)
        max_len = sent_we.shape[1]
        we_dim = sent_we.shape[-1]
        print("max-len", max_len, "type", type(max_len))
        '''
        for i in range(max_len): # 保持每一行元素值相等 TODO
            print(i, alpha[:,i])
            att_row = []
            for j in range(we_dim):
                att_row.append(alpha[:,i])
            att_row = concatenate(att_row, axis=-1)
            print("att_row", att_row)
            attention[:,i,:] = att_row
        '''

        return attention


    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0]
        print("compute_output_shape-output_shape", output_shape)
        return output_shape


class TCAContextEmbedding(layers.Layer):
    '''Context的Embedding的扩充，仅仅是数据维度的扩充'''
    def __init__(self, context_size=1, **kwargs):

        self.context_size = context_size
        super(TCAContextEmbedding, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        sent_len = input_shape[1]
        output_dim = self.context_size * input_shape[2]
        ret = (None, sent_len, output_dim)
        print("compute_output_shape-after", ret) # n*k
        return ret

    def call(self, inputs):
        '''使用padding的方式'''
        print("inputs", inputs)
        sent_we = inputs
        max_len = sent_we.shape[1]  # using the max_len
        # we_dim = sent_we.shape[2]
        # 改变计算过程
        context_wes = []
        for i in range(max_len):
            context_we = []
            semi_wind_size = int(self.context_size / 2)
            for j in reversed(range(semi_wind_size)):
                # print("mid_we", mid_we)
                first_we = sent_we[:, i - j, :] if i - j >= 0 else K.zeros_like(mid_we)
                context_we.append(first_we)
            mid_we = sent_we[:, i, :]
            context_we.append(mid_we)

            for j in range(semi_wind_size):
                last_we = sent_we[:, i + j, :] if i + j < max_len else K.zeros_like(mid_we)
                context_we.append(last_we)

            context_we = concatenate(context_we, axis=-1)  # shape=(?, 150)
            # print("context_we", context_we)
            context_we_exp = K.expand_dims(context_we, axis=1)
            # print("context_we_exp", context_we_exp)
            context_wes.append(context_we_exp)
        context_wes = concatenate(context_wes, axis=1)  # shape=(?, max_len, 150)
        print("context_wes", context_wes)  # (?, 87, 150)
        return context_wes


class TCAScoringLayer(layers.Layer):
    '''双线性的方式进行score得分'''
    def __init__(self,  **kwargs):
        super(TCAScoringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("build-input_shape", input_shape)
        outputsent_shape = input_shape[0]
        relation_we_shape = input_shape[1]
        print("build-input_shape, outputsent_shape {}, relation_we_shape {}".format(outputsent_shape, relation_we_shape))
        # Create a trainable weight variable for this layer.
        m_x_dim = outputsent_shape[-1]  # 1000
        m_y_dim = relation_we_shape[-1]  # 80
        print("m_x_dim {}, m_y_dim {}".format(m_x_dim, m_y_dim))
        self.U = self.add_weight(name='attention_probs', shape=(m_x_dim, m_y_dim), initializer='uniform',
                                 trainable=True)
        print("attention_probs", self.U)
        super(TCAScoringLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        '''score的zeta计算得分公式为(oS)⊤Ur 计算值为一个得分'''
        o_s = inputs[0]
        r = inputs[1]

        tmp_ret = K.dot(o_s, self.U)
        tmp_r = K.squeeze(r, axis=1)

        ret = tmp_ret*tmp_r # not batch dot
        print("TCAScoringLayer output", ret)
        return ret

    def compute_output_shape(self, input_shape):
        print("compute_output_shape-input_shape", input_shape)
        input_shape = (None, 1)
        return input_shape


class FirstLevelAttention(layers.Layer):
    '''第一级的attention计算方式-无参数 '''
    def __init__(self,  **kwargs):
        super(FirstLevelAttention, self).__init__(**kwargs)

    def call(self, inputs):
        '''计算出两个对角矩阵即可,根据索引的index获得实体，两个再计算内积'''
        sentence_matrix = inputs[0]
        entity_pos_index = inputs[1]

        # using slice the value with two tensor what (?, 2, 50)
        if K.dtype(entity_pos_index) != 'int32':  # need convert
            entity_pos_index = K.cast(entity_pos_index, 'int32')
        max_len = sentence_matrix.shape[1]  # using the max_len

        z = K.one_hot(entity_pos_index, max_len)
        first_att_mat = K.batch_dot(z, sentence_matrix) # To check left or right
        print("first attention-output-shape", first_att_mat)

        return first_att_mat

    def compute_output_shape(self, input_shape):
        input_0 = input_shape[0]
        input_1 = input_shape[1]
        output_shape = (None, input_1[-1], input_0[-1])
        print("output_shape", output_shape)
        return output_shape

class FirstLevelFuse(layers.Layer):
    '''第一级的attention计算方式'''
    def __init__(self,  fuse_type='avg', **kwargs):
        self.fuse_type = fuse_type
        super(FirstLevelFuse, self).__init__(**kwargs)

    def call(self, inputs):
        sentence_matrix = inputs[0]
        att_weight = inputs[1]
        # avg 的方式
        # att = 0.5*(att_weight[0]+att_weight[1])
        att = K.mean(att_weight, axis=1)
        ret = sentence_matrix*att
        print("first fuse shape", ret)
        return ret

    def compute_output_shape(self, input_shape):
        print("FirstLevelFuse-outshape", input_shape[0])
        return input_shape[0]


class SecondLevelAttention(layers.Layer):
    '''第二级attention-based-pooling的计算'''
    def __init__(self,  **kwargs):
        super(SecondLevelAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        print("build-input_shape", input_shape)
        outputsent_shape = input_shape[0]
        relation_we_shape = input_shape[1]
        print("build-input_shape, outputsent_shape {}, relation_we_shape {}".format(outputsent_shape, relation_we_shape))
        # Create a trainable weight variable for this layer.
        m_x_dim = outputsent_shape[-1]  # 1000
        m_y_dim = relation_we_shape[-1]  # 80
        print("m_x_dim {}, m_y_dim {}".format(m_x_dim, m_y_dim))
        self.U = self.add_weight(name='attention_probs', shape=(m_x_dim, m_y_dim), initializer='uniform',
                                               trainable=True)
        print("U_matrix", self.U)
        super(SecondLevelAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        '''G=trans(R_star_layer)*U*W_l -> softmax'''
        r_star_layer = inputs[0]
        weight_l = inputs[1]
        t_weight_l = K.permute_dimensions(weight_l, (0, 2, 1))

        tmp_ret = K.dot(r_star_layer, self.U)
        ret = K.batch_dot(tmp_ret, t_weight_l)
        print("SecondLevelAttention output", ret)
        return ret

    def compute_output_shape(self, input_shape):
        # (None, n, m)
        print("compute_output_shape-input_shape", input_shape)
        input_shape = (None, input_shape[0][1], 19)
        return input_shape


class SecondAttentionFuse(layers.Layer):
    '''第一级的attention计算方式'''
    def __init__(self,  fuse_type='avg', **kwargs):
        self.fuse_type = fuse_type
        super(SecondAttentionFuse, self).__init__(**kwargs)

    def call(self, inputs):
        r_star_matrix = inputs[0]
        att_weight = inputs[1]
        t_r_star_matrix = K.permute_dimensions(r_star_matrix, pattern=(0,2,1))
        ret = K.batch_dot(t_r_star_matrix, att_weight)
        print("second fuse shape", ret)
        return ret

    def compute_output_shape(self, input_shape):
        output_shape = (None, input_shape[0][-1], input_shape[1][-1])
        print("SecondAttentionFuse-outshape", output_shape)
        return output_shape


class DeltaDistanceLayer(layers.Layer):
    '''delta距离的计算'''
    def __init__(self,  **kwargs):
        super(DeltaDistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # xian norm
        w_o = inputs[0]
        w_l = inputs[1]
        w_o = K.l2_normalize(w_o)
        delata = K.cumsum(w_l-w_o) # 一个值

        return delata

    def compute_output_shape(self, input_shape):
        print("compute_output_shape-input_shape", input_shape)
        input_shape = (None, 1)
        return input_shape
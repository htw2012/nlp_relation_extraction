#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2018/3/18
"""
from keras import backend as K

from keras.callbacks import Callback
from sklearn.metrics import f1_score

import time
import os
import json


class BestCache(object):
    best_f1score_marco = 0

    def __init__(self, best_val=0.0):
        self.best_val = best_val
        super(BestCache, self).__init__()


tgt_dict = json.load(open("../data/semeval2010/semeval2010_target.dict"))
id_to_tag = {}
for k,v in tgt_dict.items():
    id_to_tag[v] = k

class F1ScoreNotOtherEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            print("logs", logs)
            y_preds = self.model.predict(self.X_val, verbose=0)

            nb_samples = y_preds.shape[0]
            # print("samples", nb_samples)
            # y_true_idx = K.argmax(self.y_val, axis=-1)

            y_true = []
            y_pred = []
            for idx in range(nb_samples):
                true_idx = self.y_val[idx].argmax()
                # print("true_idx", true_idx, "idx", idx)
                # print("type-pre", type(y_preds[idx]))
                pred_idx = y_preds[idx].argmax()
                if true_idx == 0:  # 不统计 others类型
                    # print("cur-ignore", self.y_val[idx], "pred_idx",  y_preds[idx])
                    continue
                y_true.append(true_idx)
                y_pred.append(pred_idx)
            f1score = f1_score(y_true, y_pred,  average='micro')
            # score = roc_auc_score(self.y_val, y_pred)
            logs['f1_score'] = f1score
            macro_f1score = f1_score(y_true, y_pred, average='macro')
            # print("debug--y-true-len: {:d}, y_pred-len: {:d}".format(len(y_true), nb_samples))
            print("F1-score - epoch {:d} macro_f1score {:.4f} micro_f1score {:.4f}".format(epoch + 1, macro_f1score, f1score))


class F1ScoreOfficial(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:
            print("logs", logs)
            y_preds = self.model.predict(self.X_val, verbose=0)

            nb_samples = y_preds.shape[0]

            y_true = []
            y_pred = []
            y_rel = []
            for idx in range(nb_samples):
                true_idx = self.y_val[idx].argmax()
                # print("true_idx", true_idx, "idx", idx)
                # print("type-pre", type(y_preds[idx]))
                pred_idx = y_preds[idx].argmax()

                real_relation = self.get_real_relation(pred_idx,  id_to_tag)
                y_rel.append(real_relation)

                if true_idx == 0:  # 不统计 others类型
                    # print("cur-ignore", self.y_val[idx], "pred_idx",  y_preds[idx])
                    continue
                y_true.append(true_idx)
                y_pred.append(pred_idx)

            score = self.evaluate_offical(y_rel, path="./results/current/", epoch=epoch)
            print("F1-score - epoch {:d} acc {:.4f} ".format(epoch + 1, score))

            f1score = f1_score(y_true, y_pred, average='micro')
            # score = roc_auc_score(self.y_val, y_pred)
            logs['f1_score'] = score
            macro_f1score = f1_score(y_true, y_pred, average='macro')
            # print("debug--y-true-len: {:d}, y_pred-len: {:d}".format(len(y_true), nb_samples))
            print("F1-score - epoch {:d} macro_f1score {:.4f} micro_f1score {:.4f}".format(epoch + 1, macro_f1score, f1score))


    def get_real_relation(self, pred_idx, id_to_tag):
        tag = id_to_tag.get(pred_idx)
        if tag == 'Other':
            return tag
        if tag[-2:] == '.r':
            tag = tag[:-2]+"(e2,e1)"
            # print "tag", tag
        else:
            tag = tag+"(e1,e2)"
        return tag

    def evaluate_offical(self, results, path, epoch):
        script_file = "script/semeval2010_task8_scorer-v1.2.pl"
        # 格式化成2016-03-20 11:45:39形式
        str_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        proposed_answer_file = os.path.join(path, "proposed_answer_%s.txt"%(str_time))
        answer_key_file = os.path.join("script/answer_key.txt")
        result_scores_file = os.path.join(path, "result_scores_%s.txt"%(str_time))

        with open(proposed_answer_file, "w") as f:
            for arg, line in enumerate(results):
                s_line = "{}{}{}{}".format(arg + 1, "\t", line, "\n")
                # f.write(str(arg+1)+"\t"+line+"\n")
                f.write(s_line)
        line = "perl {}  {}  {} > {}".format(script_file, proposed_answer_file, answer_key_file, result_scores_file)
        print("cwd", line)
        os.system(line)
        eval_lines = []
        with open(result_scores_file) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                # print("get-line", line.strip())
                eval_lines.append(line)
        last_line = eval_lines[-1]

        print "penult_line", eval_lines[-2]
        last_idx = last_line.find('% >>>')
        start_idx = last_line.find('F1 = ')
        score = float(float(line[start_idx+len('F1 = '):last_idx].strip())/100)
        # do some get best opt
        if score > BestCache.best_f1score_marco:
            self.record_global_results(epoch, result_scores_file, score)
        else:
            os.remove(proposed_answer_file)
            os.remove(result_scores_file)

        # print("score", score)
        return score

    def record_global_results(self, epoch, result_scores_file, score):

        # delete ori text files
        eval_folder= "./results/best_eval/"
        eval_files = os.listdir(eval_folder)
        for eval_f in eval_files:
            if eval_f == result_scores_file:
                continue
            os.remove(eval_folder+eval_f)

        # delete ori model files
        weights_folder = "./results/weights/"
        weights_files = os.listdir(weights_folder)
        for f in weights_files:
            os.remove(weights_folder+f)

        # save the current model
        BestCache.best_f1score_marco = score
        print("current global best_f1score_marco {}".format(score))

        model_filename = weights_folder + "weight_epoch-{}_f1scoremacro_{}.hdf5".format(epoch, score)
        print("save the current best model in epoch {}, with_file_name {}".format(epoch, model_filename))
        self.model.save(model_filename)


def categorical_accuracy_without_others(y_true, y_pred):
    y_true_idx = K.argmax(y_true, axis=-1)
    y_pred_idx = K.argmax(y_pred, axis=-1)

    y_true = []
    y_pred = []
    for true_idx, pred_idx in enumerate(y_true_idx, y_pred_idx):
        if true_idx == 0: # 不统计 others类型
            continue
        y_true.append(y_true_idx)
        y_pred.append(pred_idx)
    return K.cast(K.equal(y_true,y_pred_idx, K.floatx()))
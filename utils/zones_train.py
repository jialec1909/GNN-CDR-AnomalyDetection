"""
Detector Example
================

In this tutorial, you will learn the basic workflow of
PyGOD with an example of DOMINANT. This tutorial assumes that
you have basic familiarity with PyTorch and PyTorch Geometric (PyG).

(Time estimate: 5 minutes)
"""
#######################################################################
# Data Loading


import torch
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from torch_geometric.loader import DataLoader
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc
from utils import gen_outliers
from dataloader_3zones import parse_opt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def results(data_list, detector):
    for data_dic in data_list:
        test_data = data_dic['data']
        data_id = data_dic['id']
        data, y = gen_contextual_outlier(test_data, n = 100, k = 50)
        data.y = y.long()
        # data, ys = gen_structural_outlier(data, m = 10, n = 10)
        # data.y = torch.logical_or(ys, ya).long()

        print("validation results for zone %d :" %(data_id))

        pred, score, prob, conf = detector.predict(data,
                                               return_pred=True,
                                               return_score=True,
                                               return_prob=True,
                                               return_conf=True)
        print('Labels of zone %d:'%(data_id))
        print(pred)

        print('Raw scores of zone %d:'%(data_id))
        print(score)

        print('Probability of zone %d:'%(data_id))
        print(prob)

        print('Confidence of zone %d:'%(data_id))
        print(conf)
        return pred, score, prob, conf

#######################################################################
# Evaluation
# ----------
# To evaluate the performance outlier detector with AUC score, you can:



def zones_evaluation(data_list, detector):
    opt = parse_opt()
    auc_avg_eval = 0
    for data_dic in data_list:
        data = data_dic['data']
        id = data_dic['id']
        if opt.gen_outliers_method == 'manually':
            eval_data, y = gen_outliers.gen_contextual_outlier_manually(data, n=100, scale = 6)
            eval_data.y = y.long()
        else :
            eval_data, y = gen_contextual_outlier(data, n=100, k = 50)
            eval_data.y = y.long()
        # eval_data, ys = gen_structural_outlier(eval_data, m=10, n=10)
        # eval_data.y = torch.logical_or(ys, ya).long()

        pred, score, prob, conf = detector.predict(data,
                                                   return_pred = True,
                                                   return_score = True,
                                                   return_prob = True,
                                                   return_conf = True)

        min_length = min(len(eval_data.y), len(score))
        eval_data.y, score = eval_data.y[:min_length], score[:min_length]

        auc_score = eval_roc_auc(eval_data.y, score)
        print('AUC Score of zone %d:'%(id), auc_score)
        auc_avg_eval = auc_avg_eval + auc_score
    auc_avg_eval = auc_avg_eval/len(data_list)
    print('AUC Score avraged:', auc_avg_eval)
    return auc_avg_eval


def train(data_list, detector = None):
    opt = parse_opt()
    auc_avg_train = 0
    avg_train_scores = 0
    if detector is None:
        detector = DOMINANT(hid_dim = 64, num_layers = 5, epoch = 100, gpu = -1, dropout = 0.3, verbose = 2)
    else:
        detector = detector

    train_dataset = []
    for data_dic in data_list:
        data = data_dic['data']
        id = data_dic['id']
        if opt.gen_outliers_method == 'manually':
            train_data, y = gen_outliers.gen_contextual_outlier_manually(data, n=100, scale = 6)
            train_data.y = y.long()
            train_dataset.append(train_data)
        else:
            train_data, y = gen_contextual_outlier(data, n=100, k = 50)
            train_data.y = y.long()
            train_dataset.append(train_data)

    # torch.manual_seed(12345)
    # dataset = data_list['data'].shuffle()
    # train_dataset = data_list['data']
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    for step, data in enumerate(train_loader):
        print(f'step: {step +1}')
        print(f'data graphs: {data.num_graphs}')
        print(data)

    for train_data in train_loader:
        detector.fit(train_data)
        train_pred, train_scores, train_prob, train_conf =  detector.predict(train_data,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)

        auc_score = eval_roc_auc(train_data.y, train_scores)
        # print(f'Step/batch {step + 1}')
        # print("=============")
        # print(f'number of zones in this step: {train_data.num_graphs}')
        print(f'auc in this step: {auc_score}')

        auc_avg_train = auc_avg_train + auc_score
        if len(train_scores) < (batch_size * 2400):
            train_scores = torch.nn.functional.pad(train_scores, (0, (batch_size * 2400) - len(train_scores)),
                                                    value = train_scores.mean())
        print(f'train scores at this step: {train_scores}')
        avg_train_scores = avg_train_scores + train_scores

    auc_avg_train = auc_avg_train / len(train_loader)
    avg_train_scores = avg_train_scores / len(train_loader)
    print('AUC Score avraged:', auc_avg_train)

    # 用法示例

    return detector, avg_train_scores



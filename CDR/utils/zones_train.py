
#######################################################################


import torch
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from torch_geometric.loader import DataLoader
from pygod.detector import DOMINANT, CoLA
from pygod.metric import eval_roc_auc
from ..models.detector.gcnEnDe import GCNEnDe
from ..utils import gen_outliers
import wandb

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


# TODO, change global variable as entry parameter
# FIXME, look upon
def zones_evaluation(data_list, detector, gen_outliers_method: str ):
    # opt = parse_opt()
    auc_avg_eval = 0
    for data_dic in data_list:
        data = data_dic['data']
        id = data_dic['id']
        # if gengen_outlier_method:
        # if opt.gen_outliers_method == 'manually':
        if gen_outliers_method == 'manually':
            eval_data, y = gen_outliers.gen_contextual_outlier_manually(data, n=100, scale = 0.01)
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


def train(data_list, gen_outliers_method: str, train_method: str, detector = None):
    # opt = parse_opt()
    auc_avg_train = 0
    avg_train_scores = 0

    hyperparameters = dict(
        epoch = 100,
        num_layers = 1,
        gpu = -1,
        dropout = 0.1,
        verbose = 2,
        hid_dim = 64,
        outliers_gen = gen_outliers_method,
        outliers_num = 100,
        dataset = 'CDR Dataset',
        architecture = train_method
    )

    wandb.init(
        project = 'GCN-on-CDR',
        config = hyperparameters
    )


    if detector is None:
        if train_method == 'DOMINANT':
        # detector = CoLA(hid_dim = 64, num_layers = 6, epoch = 200, gpu = -1, dropout = 0.1, verbose = 2)
            detector = DOMINANT(hid_dim = hyperparameters['hid_dim'],
                                num_layers = hyperparameters['num_layers'],
                                epoch = hyperparameters['epoch'],
                                gpu = -1,
                                dropout = hyperparameters['dropout'],
                                verbose = hyperparameters['verbose'])
        elif train_method == 'GCNEnDe':
            detector = GCNEnDe(hid_dim = hyperparameters['hid_dim'],
                               num_layers = hyperparameters['num_layers'],
                                epoch = hyperparameters['epoch'],
                                gpu = -1,
                                dropout = hyperparameters['dropout'],
                                verbose = hyperparameters['verbose'])
    else:
        detector = detector

    train_dataset = []
    for data_dic in data_list:
        data = data_dic['data']
        id = data_dic['id']
        if gen_outliers_method == 'manually':
            train_data, y = gen_outliers.gen_contextual_outlier_manually(data,
                                                                         n=hyperparameters['outliers_num'],
                                                                         scale = 0.01)
            train_data.y = y.long()
            train_dataset.append(train_data)
        else:
            train_data, y = gen_contextual_outlier(data,
                                                   n=hyperparameters['outliers_num'],
                                                   k = 50)
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
        # detector.decision_score_
        train_pred, train_scores, train_prob, train_conf =  detector.predict(train_data,
                                           return_pred=True,
                                           return_score=True,
                                           return_prob=True,
                                           return_conf=True)

        auc_score = eval_roc_auc(train_data.y, train_scores)

        print(f'auc in this step: {auc_score}')

        auc_avg_train = auc_avg_train + auc_score
        if len(train_scores) < (batch_size * 2400):
            train_scores = torch.nn.functional.pad(train_scores, (0, (batch_size * 2400) - len(train_scores)),
                                                     value = train_scores.mean())
        print(f'train scores at this step: {train_scores}')
        avg_train_scores = avg_train_scores + train_scores
        # wandb.log({"train_pred": train_pred, "train_scores": train_scores,
        #            "train_prob": train_prob, "train_conf": train_conf,
        #            "auc at current step": auc_score,})

    auc_avg_train = auc_avg_train / len(train_loader)
    avg_train_scores = avg_train_scores / len(train_loader)
    print('AUC Score avraged:', auc_avg_train)
    wandb.log({'auc averaged': auc_avg_train})
    wandb.finish()


    return detector, avg_train_scores



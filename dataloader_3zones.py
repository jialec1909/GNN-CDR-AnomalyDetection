import os
import random

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader
from utils import zonesdivide, visualization, zones_train
from torch.utils.data import random_split
import argparse
import os


model_folder = './models/zones_train/'
os.makedirs(model_folder, exist_ok = True)
# data path
data_folder = './datasets/merged_datasets/merged_txt'
data_files = os.listdir(data_folder)
# save_data_obj = './datasets/data_objects'
# os.makedirs(save_data_obj, exist_ok = True)
models_files = os.listdir(model_folder)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_cell = 10000
num_zone = 100
grid_rows = 100
grid_cols = 100
cell_grid = np.arange(1, num_cell + 1).reshape((grid_rows, grid_cols))
cells_per_zone = num_cell // 100


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_method', default='DOMINANT', help='Graph based Neural Network methods, DOMINANT')
    parser.add_argument('--sample_method', default='3zones', help='Data sampling methods, 3zones or singleDay')
    parser.add_argument('--dataset_path', default='sms-call-internet-mi-2013-11-01.txt', help='dataset CDR choice (processed)')
    parser.add_argument('--save_model', default=True, help='save model')
    parser.add_argument('--zones', default='Low', help='zones for detect, High, Middle, or Low')
    parser.add_argument('--load_model', nargs='?', default=None, help='Load and Use previously trained model, e.g. model_001.pt')
    parser.add_argument('--visualization', default=False, help='show graph img')
    parser.add_argument('--gen_outliers_method', default='candidates_replace', help='the method to generate outliers, candidates_replace or manually')

    opt = parser.parse_args()
    return opt

def save_model(model, folder_path):
    model_files = [f for f in os.listdir(folder_path) if f.startswith("model_")]
    existing_count = len(model_files)
    count_formatted = f"{existing_count + 1:03}"
    model_name = f"model_{count_formatted}.pt"
    model_path = os.path.join(folder_path, model_name)
    torch.save(model, model_path)


def train(ids, dectector = None, zones = None, core_df = None, edge_index = None):
    opt = parse_opt()
    data_list = []
    for id in ids:
        zone = zones[id]
        data_z = core_df[core_df['cell_id'].isin(zone)]
        zone_data = data_z.reset_index(drop = True)
        x = torch.tensor(zone_data[['time_in_minutes', 'cell_id', 'SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet']].values, dtype = torch.float)
        neighbor_cell_pairs = edge_index[id*180: id*180+180].T
        adjacent_pairs_cell, edge_attr_cell = zonesdivide.Apairs_cell(zone_data, neighbor_cell_pairs.T)
        adjacent_pairs_cell = torch.tensor(adjacent_pairs_cell, dtype = torch.long)
        edge_attr_cell = torch.tensor(edge_attr_cell, dtype = torch.float)
        _, num_edge_cell = np.shape(adjacent_pairs_cell.T)
        adjacent_pairs_time, edge_attr_time = zonesdivide.Apairs_time(zone_data)
        adjacent_pairs_time = torch.tensor(adjacent_pairs_time, dtype = torch.long)
        edge_attr_time = torch.tensor(edge_attr_time, dtype = torch.float)

        edge_pairs = torch.cat([adjacent_pairs_cell, adjacent_pairs_time], dim = 1)
        edge_attr = [edge_attr_cell, edge_attr_time]

        data = Data(x = x, edge_index = edge_pairs)
        data.edge_attr = torch.cat(edge_attr)
        # data = data.to(device)

        # Data Object for one file saved in the list
        # data_list.append(data)
        data_dict = {'data': data, 'id': id}
        data_list.append(data_dict)

        if opt.visualization:
            visualization.visualize_graph(data, id)

    total_size = len(data_list)
    train_size = int(0.8*total_size)
    val_size = int(0.1*total_size)
    test_size = total_size - train_size - val_size
    train_data, val_data, test_data = random_split(data_list, [train_size, val_size, test_size])

    if opt.load_model is not None:
        model_path = os.path.join(model_folder, opt.load_model)
        detector = torch.load(model_path)
        # detector.eval()
        zones_train.zones_evaluation(data_list, detector)
    else:
        detector, scores = zones_train.train(train_data, detector = None)
        if opt.save_model:
            save_model(detector, model_folder)
    #results: using one data object to compute
    # pred, score, prob, conf = zones_train.results(train_data, detector)
        auc_avg_eval = zones_train.zones_evaluation(test_data, detector)
    return detector


def main():
    opt = parse_opt()
    if isinstance(opt.dataset_path, list):
        datasets_list = opt.dataset_path
    elif isinstance(opt.dataset_path, str):
        datasets_list = [opt.dataset_path]
    else:
        datasets_list = os.listdir(data_folder)
    for datasets in datasets_list:
        file_path = os.path.join(data_folder, datasets)
        df = pd.read_csv(file_path)

        zones, edge_index = zonesdivide.divide_zones(grid_rows, grid_cols, cell_grid)

        core_df = pd.DataFrame(df, columns = ["time_in_minutes", "cell_id", "SMS_in", "SMS_out", "Call_in", "Call_out",
                                              "Internet"])


        if opt.train_method == 'DOMINANT':
            if opt.sample_method == '3zones':
                sampled_time = np.arange(30, 30 + 24 * 60, 60)
                core_df = core_df[core_df['time_in_minutes'].isin(sampled_time)]
                # calculate the average activities of each zone, and rank the 100 zones, divide into 3 classes: High active, Low active, inactive
                classes_divided_results = zonesdivide.classes_zones_divide(zones, core_df)
                High_activity_ids = classes_divided_results['High_activity_zones_id']
                Middle_activity_ids = classes_divided_results['Middle_activity_zones_id']
                Low_activity_ids = classes_divided_results['Low_activity_zones_id']

                if opt.zones == 'Low':
                    detector = train(Low_activity_ids, zones = zones, core_df = core_df, edge_index = edge_index)
                elif opt.zones == 'Middle':
                    detector = train(Middle_activity_ids, zones = zones, core_df = core_df, edge_index = edge_index)
                elif opt.zones == 'High':
                    detector = train(High_activity_ids, zones = zones, core_df = core_df, edge_index = edge_index)
                else:
                    detector = train(High_activity_ids+Middle_activity_ids+Low_activity_ids, zones = zones, core_df = core_df, edge_index = edge_index)
            else:
                raise ValueError('Unknown sampling method {}'.format(opt.sample_method))
        # elif opt.method == 'sort':
        #     mot_tracker = Sort(max_age=1, min_hits=5, iou_threshold=0.3)
        else:
            raise ValueError('Unknown Anomaly Detection method {}'.format(opt.train_method))

        # detector = train(Middle_activity_ids, detector)
        # train(High_activity_ids, detector)

if __name__ == '__main__':
    main()
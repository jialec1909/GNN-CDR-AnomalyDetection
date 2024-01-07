
import torch
import numpy as np
import pandas as pd


def recons_cross_loss(x, x_):
    diff = torch.pow(x - x_, 2)
    error = torch.sqrt(torch.sum(diff, 1))
    # softmax_error = nn.functional.softmax(error, dim = -1)
    #
    # target = y.int()
    # # loss = nn.CrossEntropyLoss(softmax_error, target)
    # loss = nn.CrossEntropyLoss(error, target)

    return error



def check_missing_data(data):
    num_cell = data['cell_id'].nunique()
    if len(data) <= 144 * num_cell:
        id_lengths = data.groupby('cell_id')['time_in_minutes'].transform('size')
        missing_id = id_lengths[id_lengths != 144].index
        missing_cell_block = data.loc[data.index.isin(missing_id)]
        missing_cell_id = missing_cell_block['cell_id']
        missing_cell_id.unique().tolist()
    else:
        missing_cell_id = []
        print('No missing data in dataframe size of 1440000')
    return missing_cell_id






def padding_missing_data(data):
    missing_ids = check_missing_data(data)
    full_tpx = np.arange(0, 1440, 10)
    features_to_fill = ['SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet']

    for id in missing_ids:
        cell_data = data[data['cell_id'] == id]
        missing_tpx = sorted(set(full_tpx) - set(cell_data['time_in_minutes']))

        neighbor_data = data[data['cell_id'].isin([id - 1, id + 1])]

        for tpx in missing_tpx:
            new_row = {'time_in_minutes': tpx, 'cell_id': id}
            neighbor_values_at_time = neighbor_data[neighbor_data['time_in_minutes'] == tpx][features_to_fill]
            if not neighbor_values_at_time.empty:
                average_value = neighbor_values_at_time.mean()
                # padding
                # data = data.append({'time_in_minutes': tpx, 'cell_id': id, 'merged_activity': average_value}, ignore_index = True)
                for feature in features_to_fill:
                    new_row[feature] = average_value[feature]
            else:
                # using global mean to pad
                # global_avg_value = data['merged_activity'].mean()
                # data = data.append({'time_in_minutes': tpx, 'cell_id': id, 'merged_activity': global_avg_value}, ignore_index = True)
                for feature in features_to_fill:
                    global_avg_value = data[feature].mean()
                    new_row[feature] = global_avg_value
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

    data = data.sort_values(by = ['cell_id', 'time_in_minutes']).reset_index(drop = True)

    return data







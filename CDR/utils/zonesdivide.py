import numpy as np
import torch
import pandas as pd


def Apairs(cell_grid):
    # Find indices of adjacent pairs in rows
    rows_idx = np.argwhere((cell_grid[:, :-1] + 1) == cell_grid[:, 1:])
    adjacent_pairs_rows_node = cell_grid[rows_idx[:, 0], rows_idx[:, 1]]
    adjacent_pairs_rows = np.stack((adjacent_pairs_rows_node, adjacent_pairs_rows_node + 1), axis = -1)
    # Find indices of adjacent pairs in columns
    cols_idx = np.argwhere((cell_grid[:-1, :] + 100) == cell_grid[1:, :])
    adjacent_pairs_cols_node = cell_grid[cols_idx[:, 0], cols_idx[:, 1]]
    adjacent_pairs_cols = np.stack((adjacent_pairs_cols_node, adjacent_pairs_cols_node + 100), axis = -1)
    # Combine both rows and columns pairs
    adjacent_pairs = np.concatenate((adjacent_pairs_rows, adjacent_pairs_cols), axis = 0)
    return adjacent_pairs


def divide_zones(grid_rows, grid_cols, cell_grid):
# Split the cell grid into 100 zones (10x10)
    zones = []
    A_matrices = np.empty((0,2))
    for i in range(0, grid_rows, 10):
        for j in range(0, grid_cols, 10):
            zone = cell_grid[i:i+10, j:j+10]
            A_matrix = Apairs(zone)
            A_matrices = np.vstack((A_matrices,A_matrix))
            zone = zone.flatten()
            zones.append(zone)
    return zones, A_matrices

# calculate the average activities of each zone, and rank the 100 zones, divide into 3 classes: High active, active, low active
def rank_zones(zones, data):
    avg_activity_list = []
    # avg_sms_out_list = []
    # avg_call_in_list = []
    # avg_call_out_list = []
    # avg_internet_list = []
    for zone in zones:
        zone_data = data[data['cell_id'].isin(zone)]
        # zone_data size : 2400 * 7 (24 timepoints, 100 cells,7 features)
        avg_sms_in = np.mean(zone_data['SMS_in'])
        # avg_sms_in_list.append(avg_sms_in)
        avg_sms_out = np.mean(zone_data['SMS_out'])
        # avg_sms_out_list.append(avg_sms_out)
        avg_call_in = np.mean(zone_data['Call_in'])
        # avg_call_in_list.append(avg_call_in)
        avg_call_out = np.mean(zone_data['Call_out'])
        # avg_call_out_list.append(avg_call_out)
        avg_internet = np.mean(zone_data['Internet'])
        # avg_internet_list.append(avg_internet)
        avg_activity = (avg_call_in + avg_call_out + avg_sms_in + avg_sms_out + avg_internet / 100) / 5
        avg_activity_list.append(avg_activity)

    ranked_activity = sorted(enumerate(avg_activity_list), key = lambda x: x[1])
    return ranked_activity

def classes_zones_divide(zones, data):
    ranked_activity = rank_zones(zones, data)
    # High active: > 5
    High_active_zones = []
    # middle active: 1< activity < 5
    active_zones = []
    # low active: < 1
    low_active_zones = []

    for item in ranked_activity:
        if item[1] > 5:
            High_active_zones.append(item[0])
        elif item[1] < 1:
            active_zones.append(item[0])
        else:
            low_active_zones.append(item[0])
    divided_zones = {
        "High_activity_zones_id": High_active_zones,
        "Middle_activity_zones_id": active_zones,
        "Low_activity_zones_id": low_active_zones
    }
    return divided_zones

def Apairs_time(data):
    time_values = data['time_in_minutes'].values
    cell_ids = data['cell_id'].values

    n = len(time_values)
    time_diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            time_diff = abs(time_values[i] - time_values[j])
            time_diff_matrix[i, j] = time_diff
            time_diff_matrix[j, i] = time_diff

    # edge_indices_0 = np.argwhere(time_diff_matrix == 0)
    edge_indices_60 = np.argwhere(time_diff_matrix == 60)
    # edge_indices_0 = edge_indices_0[edge_indices_0[:, 0] != edge_indices_0[:, 1]]
    # Filter out pairs with different cell ids
    valid_pairs = []
    for edge in edge_indices_60:
        if cell_ids[edge[0]] == cell_ids[edge[1]]:
            valid_pairs.append(edge)
    # edge_indices_0 = np.array(valid_pairs)
    edge_indices_60 = np.array(valid_pairs)

    edge_indices = np.unique(np.sort(edge_indices_60, axis=1), axis=0)
    edge_attr = np.where(time_diff_matrix[edge_indices[:, 0], edge_indices[:, 1]] == 0, 0.3, 0.1)

    return edge_indices.T, edge_attr

def Apairs_cell(data, cell_pairs):
    cell_ids = data['cell_id'].values
    time_values = data['time_in_minutes'].values

    edge_indices = []
    edge_attr = []

    for cell_pair in cell_pairs:
        cell_id1, cell_id2 = cell_pair
        indices1 = np.where(cell_ids == cell_id1)[0]
        indices2 = np.where(cell_ids == cell_id2)[0]

        for i in indices1:
            for j in indices2:
                if np.abs(time_values[i] - time_values[j]) == 0:
                    edge_indices.append((i, j))
                    edge_attr.append(0.2)

    return np.array(edge_indices).T, np.array(edge_attr)


import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# data path
data_folder = './datasets/merged_datasets'
data_files = os.listdir(data_folder)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# The list is used for saving all the data objects for pyG. e.g, 7 data objects for 7 days files.
data_list = []

# look up for each data file
for file in data_files:
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)

    # extract below
    node_features = []

    # Create a 100x100 grid array with values ranging from 1 to 10000
    cell_grid = np.arange(1, 10001).reshape((100, 100))
    # Find indices of adjacent pairs in rows
    rows_idx = np.argwhere((cell_grid[:, :-1] + 1) == cell_grid[:, 1:])
    adjacent_pairs_rows_node = cell_grid[rows_idx[:,0],rows_idx[:,1]]
    adjacent_pairs_rows = np.stack((adjacent_pairs_rows_node,adjacent_pairs_rows_node+1), axis= -1)
    # Find indices of adjacent pairs in columns
    cols_idx = np.argwhere((cell_grid[:-1, :] + 100) == cell_grid[1:, :])
    adjacent_pairs_cols_node = cell_grid[cols_idx[:,0],cols_idx[:,1]]
    adjacent_pairs_cols = np.stack((adjacent_pairs_cols_node,adjacent_pairs_cols_node+100), axis= -1)
    # Combine both rows and columns pairs
    adjacent_pairs = np.concatenate((adjacent_pairs_rows, adjacent_pairs_cols),axis = 0)

    #features aggregation
    core_df = pd.DataFrame(df, columns=["time_in_minutes", "cell_id", "SMS_in", "SMS_out", "Call_in", "Call_out", "Internet"])
    def merge_to_matrix(x):
        return np.vstack((x['time_in_minutes'].to_numpy(), x['SMS_in'].to_numpy(), x['SMS_out'].to_numpy(), x['Call_in'].to_numpy(),x['Call_out'].to_numpy(),x['Internet'].to_numpy())).T

    node_features = core_df.groupby("cell_id").apply(merge_to_matrix).reset_index(name="feature_representation")
    feature_tensor = torch.tensor(np.vstack(node_features["feature_representation"]), dtype=torch.float32)
    # 将 DataFrame 转换为 NumPy 数组
    # node_features_array = node_features.to_numpy()
    # print(node_features['feature_representation'].dtypes)

    # convert node features and edge index
    # x = torch.tensor(node_features_array, dtype=torch.float32)
    edge_index = torch.tensor(adjacent_pairs-1, dtype=torch.long).t().contiguous()

    # Data object that fits pyG
    data = Data(x=feature_tensor, edge_index=edge_index)
    data = data.to(device)

    # Data Object for one file saved in the list
    data_list.append(data)

# DataLoader
batch_size = 1  # process one data file in one batch
loader = DataLoader(data_list, batch_size=batch_size)


# import torch
# import os
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# import utils.functional as functional
# from ..models.detector import transformer
# import wandb
#
# data_folder = '../datasets/merged_datasets/merged_trans'
# datasets_list = os.listdir(data_folder)
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# class CellDataset(Dataset):
#     def __init__(self, dataframe, sequence_length=6):
#         self.sequence_length = sequence_length
#         self.dataframe = dataframe
#         self.cell_ids = dataframe['cell_id'].unique()
#
#         # each 'cell_id' contains a list, which is made up of activity values.
#         # self.grouped = self.dataframe.groupby('cell_id')['merged_activity'].apply(list)
#         self.grouped = df.groupby('cell_id').apply(lambda x: x[
#             ['SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet', 'merged_activity']].values.tolist())
#     def __len__(self):
#         return len(self.cell_ids) * 24  # Each cell contains 24 sequences
#
#     def __getitem__(self, idx):
#         cell_idx = idx // 24
#         sequence_idx = (idx % 24) * self.sequence_length
#         cell_id = self.cell_ids[cell_idx]
#
#         # collect the sequence vectors
#         sequence = self.grouped[cell_id][sequence_idx:sequence_idx + self.sequence_length]
#         next_sequence = self.grouped[cell_id][sequence_idx + self.sequence_length:sequence_idx + 2 * self.sequence_length]
#
#         # 转换为 tensor
#         sequence_tensor = torch.FloatTensor(sequence)
#         next_sequence_tensor = torch.FloatTensor(next_sequence) if len(next_sequence) == self.sequence_length else torch.zeros_like(sequence_tensor)
#
#         return sequence_tensor, next_sequence_tensor
#
# def train_model(dataloader, decoder, optimizer, criterion, epochs):
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project = "Transformer_CDR",
#         config = {
#             "learning_rate": 0.001,
#             "architecture": "Transformer-Decoder",
#             "dataset": "CDR",
#             "epochs": 100,
#             "optimizer": "Adam",
#             "batch_size": 24,
#             "num_layers": 6,
#             "heads": 2,
#             "dim_k": 3,
#             "dim_v": 3,
#             "dropout": 0.0,
#             "criterion": "MSELoss",}
#     )
#
#     for batch_cell, sequence_batch in enumerate(dataloader):
#         print(f'Batch/cell: {batch_cell}')
#         input = sequence_batch[0]
#         target = sequence_batch[1]
#         src_mask = torch.ones(input.shape)
#         trg_mask = torch.ones(input.shape)
#         batch_loss = 0.0
#         for epoch in range(epochs):
#             optimizer.zero_grad()
#             out = decoder(batch_size = 24, x = input, src_mask = src_mask, trg_mask = trg_mask, y = target)
#             loss = criterion(out, target)
#             batch_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#         avg_batch_loss = batch_loss / epochs
#         print(f'---------Average batch loss for batch/cell: {avg_batch_loss}------')
#         wandb.log({'batch/cell': batch_cell, 'Average batch loss': avg_batch_loss})
#
#     wandb.finish()
# decoder = transformer.TransformerDecoder(embed_size = 6, heads = 2, dim_k = 3, dim_v = 3,
#                                          num_layers = 6, dropout = 0.0, device = device)
# criterion = nn.MSELoss()
#
# optimizer = optim.Adam(decoder.parameters(), lr=0.001)
# # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# # 假设 data 是一个 (10000, 144) 的 numpy 数组，其中 data[i] 是第 i 个 cell 的活动值
# for datasets in datasets_list:
#     file_path = os.path.join(data_folder, datasets)
#     df = pd.read_csv(file_path)
#     core_df = pd.DataFrame(df)
#     data = functional.padding_missing_data(core_df) # padding if necessary
#
#
#
#     sequence_length = 6
#     dataset = CellDataset(data, sequence_length)
#     dataloader = DataLoader(dataset, batch_size=24, shuffle=False)
# #
# #
# # -----------------------------------------------------------------------------------------
# # -------------------------------- Train --------------------------------------------------
# # -----------------------------------------------------------------------------------------
#
# #
# #
#
#     train_model(dataloader, decoder, optimizer, criterion, epochs = 100)
# #     for sequenes in dataloader:
# #         print(len(sequenes))
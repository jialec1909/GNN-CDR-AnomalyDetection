import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import CDR.utils.functional as functional
from CDR.models.detector import transformer
import wandb
import argparse

data_folder = "../CDR/datasets/merged_datasets/merged_trans"
datasets_list = os.listdir(data_folder)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', type = bool, default = True)
    # parser.add_argument('--predict', type = bool, default = False)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--sequence_length', type = int, default = 6)
    parser.add_argument('--predict_length', type = int, default = 1)
    parser.add_argument('--learning_rate', type = float, default = 0.005)
    parser.add_argument('--train_size_factor', type = float, default = 0.2)
    parser.add_argument('--test_size_factor', type = float, default = 0.1)
    parser.add_argument('--num_layers', type = int, default = 6)
    parser.add_argument('--heads', type = int, default = 8)
    parser.add_argument('--dim_k', type = int, default = 8)
    parser.add_argument('--dim_v', type = int, default = 8)
    parser.add_argument('--dropout', type = float, default = 0.0)
    parser.add_argument('--encoder_size', type = int, default = 64)
    opt = parser.parse_args()
    return opt

class CellDataset(Dataset):
    # --------------------------------------------
    # input: dataframe, sequence_length, predict_length
    # output: input_tensor, target_tensor, cell_ids

    # --------------------------------------------
    def __init__(self, dataframe, sequence_length=6, predict_length=1, train = True):
        self.sequence_length = sequence_length
        self.predict_length = predict_length
        self.train = train
        self.dataframe = dataframe
        self.cell_ids = dataframe['cell_id'].unique()

        # each 'cell_id' contains data of 144 time points.
        self.grouped = dataframe.groupby('cell_id').apply(
            lambda x: np.array(x[['SMS_in', 'SMS_out', 'Call_in','Call_out', 'Internet']]))

    def __len__(self):
        return len(self.cell_ids)  # Each cell contains (144 - sequence_length) sequences as input, and (144 - sequence_length) sequences as target.

    def __getitem__(self, cell_id):
        ## FIXME： New loader to load data for prediction
        cell_id = self.cell_ids[cell_id]
        if self.train:
            # when train,
            #       input x: (num_cell * 138, sequence_length, 5), current information
            #       label y: (num_cell * 138, sequence_length, 5), previous information
            #       label output: (num_cell * 138, predict_length, 5), to calculate loss
            input_sequences = [] # x
            target_pred = [] # output label
            for i in range(144 - self.sequence_length): # 144 - sequence_length = 138 when sequence_length = 6
                sequence_idx = i
                input_sequences.append(self.grouped[cell_id][sequence_idx:sequence_idx + self.sequence_length]) #
                if self.predict_length == 1:
                    target_pred.append(self.grouped[cell_id][sequence_idx + self.sequence_length])
                else:
                    target_pred.append(self.grouped[cell_id][sequence_idx + self.sequence_length:sequence_idx + self.sequence_length + self.predict_length])

                i += 1

            input_array = np.array(input_sequences)
            input_tensor = torch.FloatTensor(input_array)
            target_array = np.array(target_pred)
            target_tensor = torch.FloatTensor(target_array)

            input_batch_tensor = input_tensor.reshape(-1, self.sequence_length, 5)
            y_label_tensor = torch.cat((input_batch_tensor[0,:,:].unsqueeze(0), input_batch_tensor[:-1,:,:]), dim = 0)
            target_out_tensor = target_tensor.reshape(-1, self.predict_length, 5)
            cell_id = torch.tensor(cell_id)

            return input_batch_tensor, y_label_tensor, target_out_tensor, cell_id

        else:
            # when predict,
            #       input x: (num_cell * 1, sequence_length, 5),
            #       label y: None,
            #       output: (num_cell * 138, predict_length, 5)
            input_sequences = self.grouped[cell_id][0:self.sequence_length]
            target_pred = []
            for i in range(144 - self.sequence_length):
                sequence_idx = i
                if self.predict_length == 1:
                    target_pred.append(self.grouped[cell_id][sequence_idx + self.sequence_length])
                else:
                    target_pred.append(self.grouped[cell_id][sequence_idx + self.sequence_length:sequence_idx + self.sequence_length + self.predict_length])

                # output is tensor
                target_tensor = torch.FloatTensor(target_pred)

                i += 1
            input_tensor = torch.FloatTensor(input_sequences)
            input_batch_tensor = input_tensor.reshape(-1, self.sequence_length, 5)
            target_batch_tensor = target_tensor.reshape(-1, self.predict_length, 5)
            cell_id = torch.tensor(cell_id)
            return input_batch_tensor, target_batch_tensor, cell_id

def train_collate_fn(batch):
    x, y, out, cell_idx = zip(*batch)
    x_batch = torch.stack(x)
    y_batch = torch.stack(y)
    out_batch = torch.stack(out)
    cell_idx = torch.stack(cell_idx)
    return x_batch, y_batch, out_batch, cell_idx

def test_collate_fn(batch):
    x, out, cell_idx = zip(*batch)
    x_batch = torch.stack(x)
    out_batch = torch.stack(out)
    cell_idx = torch.stack(cell_idx)
    return x_batch, out_batch, cell_idx

def train_model(train_dataloader, test_dataloader, decoder, optimizer, criterion, epochs, batch_size, train_size, test_size, learning_rate,
                num_layers, heads, dim_k, dim_v, dropout, encoder_size, device, sequence_length = 6, predict_length = 1):
    wandb.init(
        project = "Transformer_CDR",
        config = {
            "learning_rate": learning_rate,
            "architecture": "Transformer-Decoder",
            "dataset": "CDR",
            "Train_size": train_size,
            "Test_size": test_size,
            "epochs": epochs,
            "optimizer": "Adam",
            "batch_size": batch_size,
            "num_layers": num_layers,
            "heads": heads,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "dropout": dropout,
            "sequence_length": sequence_length,
            "predict_length": predict_length,
            "encoder_size": encoder_size,
            "criterion": "MSELoss", }
    )
    wandb.watch(decoder)
    step = 0
    print(f'---------------------------Training dataset-------------------------------------')
    for epoch in range(epochs):
        print(f'---------Epoch: {epoch}------')
        epoch_loss = 0.0
        for batch_num, batch_cells in enumerate(train_dataloader):
            step += 1
            cell_idx = batch_cells[3]
            num_cells = len(cell_idx)
            x_batch = batch_cells[0] # current information shape (num_cells * 138, sequence_length, 5)
            x_batch = x_batch.to(device)
            y_batch = batch_cells[1] # previous information shape (num_cells * 138, sequence_length, 5)
            y_batch = y_batch.to(device)
            out_batch = batch_cells[2] # output label shape (num_cells * 138, predict_length, 5)
            out_batch = out_batch.to(device)
            print(f'Batch: {batch_num}')
            print(f'Cells at the batch: {cell_idx.tolist()}')

            input = x_batch.view(-1, sequence_length, 5)
            y = y_batch.view(-1, sequence_length, 5)
            out_label = out_batch.view(-1, predict_length, 5)

            src_mask = torch.ones(input.shape)
            trg_mask = torch.ones(input.shape)

            optimizer.zero_grad()

            out = decoder(batch_size = num_cells * (144 - sequence_length), x = input, src_mask = src_mask, trg_mask = trg_mask, y = y)
            loss = criterion(out, out_label)
            print(f'Loss of batch {batch_num}:  {loss.item()}')
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            wandb.log({'batch': batch_num, 'batch loss': loss.item(), 't': step})
        avg_batch_loss = epoch_loss / len(train_dataloader)
        wandb.log({'epoch': epoch, 'Average epoch loss': avg_batch_loss})
        print(f'---------Epoch: {epoch}------, average loss of the dataset: {avg_batch_loss}------')

    print(f'---------------------------Training ends for the batch-------------------------------------')
    print(f'---------------------------Testing prediction begins-------------------------------------')
    for batch in test_dataloader:
        cell_idx = batch[2]
        num_cells = len(cell_idx)
        x_batch = batch[0]
        x_batch = x_batch.to(device)
        out_batch = batch[1]
        out_batch = out_batch.to(device)
        print(f'Cells at the batch: {cell_idx.tolist()}')

        input = x_batch.view(-1, sequence_length, 5)
        out_target = out_batch.view(-1, predict_length, 5)

        src_mask = torch.ones(input.shape)
        trg_mask = torch.ones(input.shape)

        out = decoder(batch_size = num_cells, x = input, src_mask = src_mask, trg_mask = trg_mask, y = None)
        predict_loss = criterion(out, out_target)
        print(f'Prediction Loss of batch {batch_num}:  {predict_loss.item()}')
        wandb.log({'test_batch': batch_num, 'test_batch loss': predict_loss.item(), 'test_t': step})

    wandb.finish()

def main():
    # -----------------------------------------------------------------------------------------
    opt = parse_args()
    criterion = nn.MSELoss()
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size
    sequence_length = opt.sequence_length
    predict_length = opt.predict_length
    train_size_factor = opt.train_size_factor
    test_size_factor = opt.test_size_factor
    num_layers = opt.num_layers
    heads = opt.heads
    dim_k = opt.dim_k
    dim_v = opt.dim_v
    dropout = opt.dropout
    epochs = opt.epochs
    encoder_size = opt.encoder_size


    decoder = transformer.TransformerDecoder(embed_size = 5 , encoding_size = encoder_size, heads = heads, dim_k = dim_k, dim_v = dim_v,
                                             sequence_length = sequence_length,
                                             predict_length = predict_length,
                                             num_layers = num_layers, dropout = dropout, device = device)
    decoder.to(device)

    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)

    for datasets in datasets_list:
        file_path = os.path.join(data_folder, datasets)
        print(f'---------------------------Loading dataset: {datasets}-------------------------------------')
        df = pd.read_csv(file_path)
        core_df = pd.DataFrame(df)
        data = functional.padding_missing_data(core_df)  # padding if necessary
        # data shape -> (10000* 144, 8) ，data[i] -> i-th cell data

        train_sets = CellDataset(data, sequence_length, predict_length, train = True)
        test_sets = CellDataset(data, sequence_length, predict_length, train = False)

        train_size = int(train_size_factor * len(train_sets))
        test_size = int(test_size_factor * len(test_sets))

        train_dataset, _ = random_split(train_sets, [train_size, (len(train_sets) - train_size)])
        _, test_dataset = random_split(test_sets, [(len(test_sets) - test_size), test_size])

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                                  collate_fn = train_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True,
                                 collate_fn = test_collate_fn)

        # # -----------------------------------------------------------------------------------------
        # # -------------------------------- Train --------------------------------------------------
        # # -----------------------------------------------------------------------------------------
        print(f'---------------------------Training dataset: {datasets}-------------------------------------')
        train_model(train_loader, test_loader, decoder, optimizer, criterion, epochs = epochs, batch_size = batch_size,
                    train_size = train_size_factor, test_size = test_size_factor, learning_rate = learning_rate,
                    num_layers = num_layers, heads = heads, dim_k = dim_k, dim_v = dim_v, dropout = dropout,
                    encoder_size = encoder_size, device = device, sequence_length = 6, predict_length = 1)




    return

if __name__ == '__main__':
    main()


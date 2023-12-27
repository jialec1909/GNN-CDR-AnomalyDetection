import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import CDR.utils.functional as functional
from CDR.models.detector import transformer
import wandb

data_folder = '../datasets/merged_datasets/merged_trans'
datasets_list = os.listdir(data_folder)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CellDataset(Dataset):
    # --------------------------------------------
    # input: dataframe, sequence_length
    # output: sequence_tensor, next_sequence_tensor

    # --------------------------------------------
    def __init__(self, dataframe, sequence_length=6):
        self.sequence_length = sequence_length
        self.dataframe = dataframe
        self.cell_ids = dataframe['cell_id'].unique()

        # each 'cell_id' contains a list, which is made up of activity values.
        # self.grouped = self.dataframe.groupby('cell_id')['merged_activity'].apply(list)
        self.grouped = dataframe.groupby('cell_id').apply(lambda x: x[
            ['SMS_in', 'SMS_out', 'Call_in', 'Call_out', 'Internet', 'merged_activity']].values.tolist())

    def __len__(self):
        return len(self.cell_ids)  # Each cell contains 24 sequences

    def __getitem__(self, cell_id):
        cell_id = self.cell_ids[cell_id]

        sequences = []
        next_sequences = []

        for i in range(24):
            sequence_idx = i * self.sequence_length
            sequence = self.grouped[cell_id][sequence_idx:sequence_idx + self.sequence_length]
            next_sequence = self.grouped[cell_id][
                            sequence_idx + self.sequence_length:sequence_idx + 2 * self.sequence_length]

            # output is tensor
            sequence_tensor = torch.FloatTensor(sequence)
            next_sequence_tensor = torch.FloatTensor(next_sequence) if len(
                next_sequence) == self.sequence_length else torch.zeros_like(sequence_tensor)
            sequences.append(sequence_tensor)
            next_sequences.append(next_sequence_tensor)

        sequence_batch_tensor = torch.stack(sequences).reshape(24, 6, 6)
        next_sequence_batch_tensor = torch.stack(next_sequences).reshape(24, 6, 6)
        cell_id = torch.tensor(cell_id)
        return sequence_batch_tensor, next_sequence_batch_tensor, cell_id


def train_model(dataloader, decoder, optimizer, criterion, epochs, batch_size, train_size, learning_rate):
    wandb.init(
        project = "Transformer_CDR",
        config = {
            "learning_rate": learning_rate,
            "architecture": "Transformer-Decoder",
            "dataset": "CDR",
            "Train_size": train_size,
            "epochs": epochs,
            "optimizer": "Adam",
            "batch_size": batch_size,
            "num_layers": 6,
            "heads": 2,
            "dim_k": 3,
            "dim_v": 3,
            "dropout": 0.0,
            "criterion": "MSELoss", }
    )
    wandb.watch(decoder)
    step = 0
    for epoch in range(epochs):
        print(f'---------Epoch: {epoch}------')
        epoch_loss = 0.0
        for batch_num, batch_cells in enumerate(dataloader):
            step += 1
            cell_idx = batch_cells[2]
            num_cells = len(cell_idx)
            sequence_batch = batch_cells[0]
            next_sequence_batch = batch_cells[1]
            print(f'Batch: {batch_num}')
            print(f'Cells at the batch: {cell_idx.tolist()}')

            input = sequence_batch.view(-1, 6, 6)
            target = next_sequence_batch.view(-1, 6, 6)

            src_mask = torch.ones(input.shape)
            trg_mask = torch.ones(input.shape)

            loss_mask = torch.ones(num_cells, 24, dtype = torch.bool)
            loss_mask[:, 23] = False
            loss_mask = loss_mask.view(-1)

            target_cut = target[loss_mask]  # cut the last sequence(23) of each cell,
            # because the value is 0, means nothing when calculating loss.

            optimizer.zero_grad()

            out = decoder(batch_size = num_cells * 24, x = input, src_mask = src_mask, trg_mask = trg_mask, y = target)
            out_cut = out[loss_mask]
            loss = criterion(out_cut, target_cut)
            print(f'Loss of batch {batch_num}:  {loss.item()}')
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            wandb.log({'batch': batch_num, 'batch loss': loss.item(), 't': step})
        avg_batch_loss = epoch_loss / len(dataloader)
        wandb.log({'epoch': epoch, 'Average epoch loss': avg_batch_loss})
        print(f'---------Epoch: {epoch}------, average loss of the dataset: {avg_batch_loss}------')

    wandb.finish()

def main():
    # -----------------------------------------------------------------------------------------
    decoder = transformer.TransformerDecoder(embed_size = 6, heads = 2, dim_k = 3, dim_v = 3,
                                             num_layers = 6, dropout = 0.0, device = device)

    criterion = nn.MSELoss()
    learning_rate = 0.005
    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    batch_size = 32

    seed = 42
    torch.manual_seed(seed)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    def custom_collate_fn(batch):
        sequence, next_sequence, cell_idx = zip(*batch)
        sequence_batch = torch.stack(sequence)
        next_sequence_batch = torch.stack(next_sequence)
        cell_idx = torch.stack(cell_idx)
        return sequence_batch, next_sequence_batch, cell_idx

    for datasets in datasets_list:
        file_path = os.path.join(data_folder, datasets)
        df = pd.read_csv(file_path)
        core_df = pd.DataFrame(df)
        data = functional.padding_missing_data(core_df)  # padding if necessary
        # data shape -> (10000* 144, 8) ，data[i] -> i-th cell data
        sequence_length = 6
        dataset = CellDataset(data, sequence_length)

        train_size_factor = 0.6
        train_size = int(train_size_factor * len(dataset))  # test train for convergence
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                                  collate_fn = custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, collate_fn = custom_collate_fn)
        # dataloader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn = custom_collate_fn)
        # #
        # # -----------------------------------------------------------------------------------------
        # # -------------------------------- Train --------------------------------------------------
        # # -----------------------------------------------------------------------------------------
        #
        train_model(train_loader, decoder, optimizer, criterion, epochs = 400, batch_size = batch_size, train_size = train_size_factor, learning_rate = learning_rate)
    # #     for sequenes in dataloader:
    # #         print(len(sequenes))
    return

if __name__ == '__main__':
    main()


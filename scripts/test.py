import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import CDR.utils.functional as functional
import CDR.utils.gen_outliers as gen_outliers
from CDR.models.detector import transformer_test
import wandb
import argparse
import matplotlib.pyplot as plt
import CDR
import random
from sklearn.metrics import roc_auc_score

CDR_path = CDR.__path__[0]
data_folder = f'{CDR_path}/../datasets/merged_datasets/merged_trans'
datasets_list = os.listdir(data_folder)

out_dir = "./runs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

existing_runs = [d for d in os.listdir (out_dir) if
                 os.path.isdir (os.path.join (out_dir, d)) and d.startswith ('run')]
existing_run_numbers = [int (run[3:]) for run in existing_runs if run[3:].isdigit ()]

if existing_run_numbers:
    new_run_no = max (existing_run_numbers) + 1
else:
    new_run_no = 1

new_run_dir = os.path.join (out_dir, f'run{new_run_no}')
os.makedirs (new_run_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', type = bool, default = True)
    # parser.add_argument('--predict', type = bool, default = False)
    parser.add_argument('--epochs', type = int, default = 700)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--sequence_length', type = int, default = 6)
    parser.add_argument('--predict_length', type = int, default = 1)
    parser.add_argument('--initial_lr', type = float, default = 1.42)
    parser.add_argument('--train_size_factor', type = float, default = 0.9995)
    # parser.add_argument('--test_size_factor', type = float, default = 0.001)
    parser.add_argument('--num_layers', type = int, default = 6)
    parser.add_argument('--heads', type = int, default = 8)
    parser.add_argument('--dim_k', type = int, default = 8)
    parser.add_argument('--dim_v', type = int, default = 8)
    parser.add_argument('--dropout', type = float, default = 0.0)
    parser.add_argument('--encoder_size', type = int, default = 64)
    parser.add_argument('--if_write', type = bool, default = True)
    parser.add_argument('--pe', type = str, default = 'hybrid_sine')
    parser.add_argument('--MHA', type = int, default = 1)
    parser.add_argument('--pe_method', type = str, default = 'original')
    opt = parser.parse_args()
    return opt

class CellDataset(Dataset):
    # --------------------------------------------
    # input: dataframe, sequence_length, predict_length
    # output: input_tensor, target_tensor, cell_ids

    # --------------------------------------------
    def __init__(self, dataframe, sequence_length = 6):
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.cell_ids = dataframe['cell_id'].unique()

        # each 'cell_id' contains data of 144 time points.
        self.grouped = dataframe.groupby('cell_id').apply(
            lambda x: np.array(x[['SMS_in', 'SMS_out', 'Call_in','Call_out', 'Internet']]))

    def __len__(self):
        return len(self.cell_ids)  # Each cell contains 144 sequences as input.

    def __getitem__(self, cell_id):

        cell_id = self.cell_ids[cell_id]

        # input x: (num_cell, 144, 5), whole sequence for the cell i
        input_sequences = []  # x
        input_sequences.append (self.grouped[cell_id])

        input_array = np.array (input_sequences)
        input_tensor = torch.FloatTensor (input_array)


        input_batch_tensor = input_tensor.reshape (-1, 144, 5)
        cell_id = torch.tensor (cell_id)

        return input_batch_tensor, cell_id


def lambda_lr(step_num, d_model, warmup_steps=5000):
    return np.power(d_model, -0.5) * np.min([
        np.power(step_num, -0.5),
        step_num * np.power(warmup_steps, -1.5)])

def test_collate_fn(batch):
    # with outliers label mixed in
    x, cell_idx, labels = zip(*batch)
    x_batch = torch.stack(x)
    cell_idx = torch.stack(cell_idx)
    labels_batch = torch.stack(labels)
    return x_batch, cell_idx, labels_batch

def collate_fn(batch):
    x, cell_idx = zip(*batch)
    x_batch = torch.stack(x)
    cell_idx = torch.stack(cell_idx)
    return x_batch, cell_idx

def future_mask(size):
    # upper-triangular matrix, upper right corner is True (to mask).
    mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()
    return mask

def tensors_to_csv(tensor1, tensor2, output_dir, cell_id, reshape_dim = (143, 5), headers = ['prediction', 'target']):

    df1 = pd.DataFrame(tensor1.cpu().detach().numpy().reshape(reshape_dim))
    df1.columns = [f'{headers[0]}_{i}' for i in range(df1.shape[1])]

    df2 = pd.DataFrame(tensor2.cpu().detach().numpy().reshape(reshape_dim))
    df2.columns = [f'{headers[1]}_{i}' for i in range(df2.shape[1])]

    combined_df = pd.concat([df1, df2], axis=1)
    file_name = os.path.join (output_dir, f'prediction_comparison of cell {cell_id}.csv')
    combined_df.to_csv(file_name, index=False)
    plt.figure(figsize = (15, 12))

    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(df1.index, df1.iloc[:, i], label = f'{headers[0]}_{i}', marker = 'o')
        plt.plot(df2.index, df2.iloc[:, i], label = f'{headers[1]}_{i}', marker = 'x')
        plt.title(f'Comparison of {headers[0]}_{i} and {headers[1]}_{i}')
        plt.xlabel('time_step')
        plt.ylabel('Values')
        plt.legend()

    plt.tight_layout()
    image_file_path = os.path.join (output_dir, f'prediction_and_target_in_cell_{cell_id}.png')
    plt.savefig (image_file_path)
    plt.close()

def train_model(train_dataloader, test_dataloader, decoder, optimizer, scheduler, criterion, test_criterion, epochs, batch_size, train_size, test_size, learning_rate,
                num_layers, heads, dim_k, dim_v, dropout, encoder_size, device, if_write, pe, MHA, pe_method, sequence_length = 6, predict_length = 1):
    wandb.init(
        project = "Transformer_CDR",
        config = {
            "learning_rate": scheduler.get_last_lr()[0],
            "architecture": "Transformer-Decoder",
            "dataset": "CDR",
            "Train_size": train_size,
            "Test_size": test_size,
            "epochs": epochs,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "batch_size": batch_size,
            "num_layers": num_layers,
            "heads": heads,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "dropout": dropout,
            "sequence_length": sequence_length,
            "predict_length": predict_length,
            "encoder_size": encoder_size,
            "criterion": "MSELoss",
            "PE": pe,
            "MHA": MHA,
            "pe_method": pe_method}
    )
    wandb.watch(decoder)
    step = 0
    print(f'---------------------------Training dataset-------------------------------------')
    for epoch in range(epochs):
        print(f'---------Epoch: {epoch}------')
        epoch_loss = 0.0
        for batch_num, batch_cells in enumerate(train_dataloader):
            step += 1
            cell_idx = batch_cells[1]
            num_cells = len(cell_idx)
            x_batch = batch_cells[0] # current information shape (num_cells, 144, 5)
            input = x_batch.view(-1, 144, 5).to(device)
            print(f'Batch: {batch_num}')
            print(f'Cells at the batch: {cell_idx.tolist()}')

            input_mask = future_mask(input.shape[1]).unsqueeze(0).to(device)

            optimizer.zero_grad()
            out = decoder(batch_size = num_cells, x = input, future_mask = input_mask, pe = pe, MHA = MHA, pe_method = pe_method, status = 'train')
            out_loss = out[:, :-1, :]
            label_loss = input[:, 1:, :]

            loss = criterion(out_loss, label_loss)
            print(f'Loss of batch {batch_num}:  {loss.item()}')
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            wandb.log({'batch': batch_num, 'batch loss': loss.item(), 't': step})
        avg_batch_loss = epoch_loss / len(train_dataloader)
        wandb.log({'epoch': epoch, 'Average epoch loss': avg_batch_loss})
        print(f'---------Epoch: {epoch}------, average loss of the dataset: {avg_batch_loss}------')

                ## Test Loss for each 100 epochs of training
        if (epoch - 1) % 50 == 0:
            test_step = (epoch - 1) // 50
            test_batch_num = 0
            test_total_auc = 0.0
            test_total_loss = 0.0
            ## TODO: fit the shape of new batch inputs with outliers.
            for batch in test_dataloader:
                cell_idx = batch[1]
                x_batch = batch[0]
                permute_labels = batch[2] # outliers positions (b, t, f)
                x_batch = x_batch.view(-1, 144, 5).to(device)
                label = x_batch[:, 1:, :].to(device) # (b, t-1, f) to align with predictions.
                permute_labels = permute_labels[:, 1:, :].to(device) # (b, t-1, f) to align with predictions.
                print(f'Cells at the batch: {cell_idx.tolist()}')
                input_mask = future_mask(x_batch.shape[1]).unsqueeze(0).to(device)

                test_batch_loss = 0.0
                test_batch_auc = 0.0
                test_num_cells = len(cell_idx)
                for id, cell in enumerate(cell_idx):
                    input = x_batch[id].unsqueeze(0).to(device)
                    out = decoder(batch_size = 1, x = input, future_mask = input_mask, pe = pe, MHA = MHA, pe_method = pe_method, status = 'predict')
                    out_loss = out[:, :-1, :]
                    label_batch = label[id]
                    if (epoch + 1) == epochs:
                        if if_write:
                            tensors_to_csv(out_loss, label_batch, output_dir = new_run_dir, cell_id = cell)
                    loss = criterion(out_loss.squeeze(0), label_batch)
                    scores = test_criterion(out_loss.squeeze(0), label_batch) # (t-1, f)
                    all_labels = permute_labels[id].view(-1)
                    all_scores = scores.view(-1)
                    auc_cell = roc_auc_score(all_labels.cpu().detach().numpy(), all_scores.cpu().detach().numpy())
                    print(f'Test Cell: {cell}, AUC: {auc_cell}')
                    test_batch_auc += auc_cell
                    test_batch_loss += loss.item()
                test_batch_num += test_num_cells
                test_total_loss += test_batch_loss
                test_total_auc += test_batch_auc

                    #wandb.log({'cell_id': cell, 'prediction': out, 'target': label})
                wandb.log ({"Test Loss": test_batch_loss / test_num_cells, "Test cell AUC": test_batch_auc / test_num_cells})
            wandb.log({'Average Test Loss': test_total_loss/test_batch_num, 'Test Step': test_step, 'Test AUC': test_total_auc/test_batch_num})


    print(f'---------------------------Training ends for the dataset-------------------------------------')


    wandb.finish()

def main():
    # -----------------------------------------------------------------------------------------
    opt = parse_args()
    criterion = nn.MSELoss()
    test_criterion = nn.MSELoss(reduction='none')
    initial_lr = opt.initial_lr
    batch_size = opt.batch_size
    sequence_length = opt.sequence_length
    predict_length = opt.predict_length
    train_size_factor = opt.train_size_factor
    test_size_factor = 1 - train_size_factor
    num_layers = opt.num_layers
    heads = opt.heads
    dim_k = opt.dim_k
    dim_v = opt.dim_v
    dropout = opt.dropout
    epochs = opt.epochs
    encoder_size = opt.encoder_size
    if_write = opt.if_write
    pe = opt.pe
    MHA = opt.MHA
    pe_method = opt.pe_method

    np.random.seed(123)
    torch.manual_seed(123)
    random.seed(123)


    decoder = transformer_test.TransformerDecoder(embed_size = 5 , encoding_size = encoder_size, heads = heads, dim_k = dim_k, dim_v = dim_v,
                                             sequence_length = sequence_length,
                                             predict_length = predict_length,
                                             num_layers = num_layers, dropout = dropout, device = device)
    decoder.to(device)

    # optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    optimizer = torch.optim.Adam (decoder.parameters (), lr = initial_lr, betas = (0.9, 0.98), eps = 1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR (optimizer, lr_lambda = lambda step_num: lambda_lr (step_num, encoder_size, 50000))

    # for datasets in datasets_list:

    selected_dataset_index = 0
    dataset = datasets_list[selected_dataset_index]
    file_path = os.path.join(data_folder, dataset)
    print(f'---------------------------Loading dataset: {dataset}-------------------------------------')
    df = pd.read_csv(file_path)
    core_df = pd.DataFrame(df)
    data = functional.padding_missing_data(core_df)  # padding if necessary
    # data shape -> (10000* 144, 8) ，data[i] -> i-th cell data

    sets = CellDataset(data, sequence_length = sequence_length)

    train_size = int(train_size_factor * len(sets))
    test_size = len(sets) - train_size

    train_dataset, test_dataset = random_split(sets, [train_size, test_size])

    test_data = torch.stack([test_dataset.dataset[i][0] for i in test_dataset.indices]).squeeze(1) # (b, t, f)

    permuted_data, outlier_label = gen_outliers.gen_transformer_outlier_manually(test_data, scale = 0.5, outlier_fraction = 0.1)
    permuted_test_dataset = TensorDataset(permuted_data, torch.tensor(test_dataset.indices), outlier_label)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                                  collate_fn = collate_fn)
    test_loader = DataLoader(permuted_test_dataset, batch_size = batch_size, shuffle = False,
                                    collate_fn = test_collate_fn)

    # # -----------------------------------------------------------------------------------------
    # # -------------------------------- Train --------------------------------------------------
    # # -----------------------------------------------------------------------------------------
    print(f'---------------------------Training dataset: {dataset}-------------------------------------')
    train_model(train_loader, test_loader, decoder, optimizer, scheduler, criterion, test_criterion, epochs = epochs, batch_size = batch_size,
                    train_size = train_size_factor, test_size = test_size_factor, learning_rate = initial_lr,
                    num_layers = num_layers, heads = heads, dim_k = dim_k, dim_v = dim_v, dropout = dropout,
                    encoder_size = encoder_size, device = device, if_write = if_write, pe = pe, MHA = MHA, pe_method = pe_method,
                sequence_length = sequence_length, predict_length = predict_length)


    return

if __name__ == '__main__':
    main()


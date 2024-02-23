import numpy as np
import matplotlib.pyplot as plt
from CDR.models.detector import transformer

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def future_mask(size):
    # upper-triangular matrix, upper right corner is True (to mask).
    mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()
    return mask

def generate_data(num_sequences=100, seq_length=144, num_features=5):
    # 初始化数据数组
    data = np.zeros ((num_sequences, seq_length, num_features))

    # 时间步
    timesteps = np.linspace (0, 2 * np.pi, seq_length)

    # 生成数据
    for i in range (num_sequences):
        for f in range (num_features):
            if f % 5 == 0:
                # 正弦波
                data[i, :, f] = np.sin (timesteps + np.random.rand () * 2 * np.pi)
            elif f % 5 == 1:
                # 余弦波
                data[i, :, f] = np.cos (timesteps + np.random.rand () * 2 * np.pi)
            elif f % 5 == 2:
                # 线性增长
                data[i, :, f] = np.linspace (0, 1, seq_length) + np.random.rand (seq_length) * 0.2
            elif f % 5 == 3:
                # 随机噪声
                data[i, :, f] = np.random.rand (seq_length)
            else:
                # 正弦波与余弦波的结合
                data[i, :, f] = 0.5 * np.sin (timesteps) + 0.5 * np.cos (timesteps + np.random.rand () * 2 * np.pi)
    data = torch.tensor (data, dtype=torch.float32)
    return data

def train_model(decoder, optimizer, train_dataloader, epochs, device, criterion):
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_id, batch in enumerate(train_dataloader):
            print(f"Epoch {epoch}, Batch {batch_id}")
            num_cells = batch.shape[0]
            input = batch.to(device)
            input_mask = future_mask(input.shape[1]).unsqueeze(0).to(device)
            optimizer.zero_grad ()
            out = decoder (batch_size = num_cells, x = input, future_mask = input_mask, status = 'train')
            out_loss = out[:, :-1, :]
            label_loss = input[:, 1:, :]
            loss = criterion (out_loss, label_loss)
            print (f'Loss of batch {batch_id}:  {loss.item ()}')
            epoch_loss += loss.item ()
            loss.backward ()
            optimizer.step ()
        avg_batch_loss = epoch_loss / len (train_dataloader)
        print (f'Epoch {epoch} average loss: {avg_batch_loss}')
    return decoder

def predict_model(model, test_dataloader, device):

    predictions = []
    with torch.no_grad ():
        for batch_id, batch in enumerate (test_dataloader):
            print (f"Batch {batch_id}, Predicting...")
            num_cells = batch.shape[0]
            input = batch.to (device)
            input_mask = future_mask (input.shape[1]).unsqueeze (0).to (device)
            out = model (batch_size = num_cells, x = input, future_mask = input_mask, status = 'predict')
            predictions.append (out.cpu ().numpy ())
    return np.concatenate (predictions), test_dataloader.dataset.indices

# 生成数据集
dataset = generate_data ()
train_size = int (0.95 * len(dataset))
test_size = len (dataset) - train_size

train_dataset, test_dataset = random_split (dataset, [train_size, test_size])


# 创建数据加载器
train_loader = DataLoader (train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader (test_dataset, batch_size = 5, shuffle = True)

decoder = transformer.TransformerDecoder(embed_size = 5, encoding_size = 64, heads = 8, dim_k = 8,
                                        dim_v = 8,
                                        sequence_length = 144,
                                        predict_length = 1,
                                        num_layers = 6, dropout = 0.00, device = device)
decoder.to (device)

optimizer = optim.Adam (decoder.parameters (), lr = 0.003)
criterion = torch.nn.MSELoss ()

model = train_model (decoder, optimizer, train_loader,300, device, criterion)
predictions, indices = predict_model(model, test_loader, device)
print('shape of predictions:', predictions.shape)

# 定义颜色
original_colors = ['green', 'lime', 'olive', 'darkgreen', 'lightgreen']  # 原始数据的绿色系列
predicted_colors = ['red', 'salmon', 'darkred', 'crimson', 'lightcoral']  # 预测数据的红色系列


test_data = np.array([dataset[i] for i in test_dataset.indices])
for seq_idx in range(len(test_data)):
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15), sharex=True)
    fig.suptitle(f"Sequence {seq_idx+1} Features Comparison")

    # 绘制每个特征轨迹
    for feature_idx in range(5):
        axes[feature_idx].plot(test_data[seq_idx, :, feature_idx], color=original_colors[feature_idx], label='Original Feature {}'.format(feature_idx+1))
        axes[feature_idx].plot(predictions[seq_idx, :, feature_idx], color=predicted_colors[feature_idx], label='Predicted Feature {}'.format(feature_idx+1))
        axes[feature_idx].legend()

    plt.xlabel("Time Step")
    plt.show()
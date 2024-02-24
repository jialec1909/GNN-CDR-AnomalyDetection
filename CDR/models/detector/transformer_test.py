import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, D, dropout=0.0, N = 144):
        super(PositionalEncoding, self).__init__()
        # D: the dimension of the embedding.
        # dropout: the dropout value (default=0.0).
        # max_len: the maximum length of the incoming sequence (default=6).
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(1, N, D)
        # pe: the positional encodings (batch_size, N, D).
        # batch_size=1 means pe works same on all batches.
        position = torch.arange(0, N, dtype=torch.float32).unsqueeze(1)
        # position: the positions values computation matrix numerator (N, 1).
        div_term = torch.pow(10000, torch.arange(0, D, 2, dtype = torch.float32) / D)
        # div_term: the denominator div_term (D/2,).
        pe[:,:, 0::2] = torch.sin(position * div_term)
        # pe[:, 0::2]: the even index of pe (N, D/2).
        if D % 2 == 1:
            pe[:, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:,:, 1::2] = torch.cos(position * div_term)
        # pe[:, 1::2]: the odd index of pe (N, D/2).
        self.register_buffer('pe', pe)
        # self.pe: the positional encodings (1, N, D).

    def forward(self, X):
        # x: the input sequence (batch_size, sequence_length, D).
        X = X + self.pe[:, :X.shape[1], :].to(X.device)
        # Broadcasting: self.P[:, :X.shape[1], :].to(X.device) (1, X.shape[1], D) -> (batch_size, X.shape[1], D).
        return self.dropout(X)
        # return: the output sequence (batch_size, sequence_length, D).


class Additive_PositionalEncoding(nn.Module):
    def __init__(self, D = 5, N = 144):
        super(Additive_PositionalEncoding, self).__init__()
        pe = torch.arange(0, N, dtype=torch.float32).unsqueeze(1) # (N, 1)
        self.pe = pe / torch.max(pe)
        self.D = D

    def forward(self, X): # X: (b, n, D)
        PE_expanded = self.pe.expand (-1, X.shape[0]).unsqueeze (2).transpose (0, 1)
        PE = PE_expanded.to (X.device)
        X = torch.cat ((X, PE), dim = 2)  # (b, 144, D+1)
        return X


class MultiHeadAttention(nn.Module):
    def __init__(self, dimension_emb, heads, dim_k, dim_v):
        super(MultiHeadAttention, self).__init__()
        self.D = dimension_emb
        self.h = heads
        self.d_k = dim_k
        self.d_v = dim_v

        self.linear_values = nn.Linear(self.D, self.h * self.d_v, bias = False)
        self.linear_keys = nn.Linear(self.D, self.h * self.d_k, bias = False)
        self.linear_queries = nn.Linear(self.D, self.h * self.d_k, bias = False)
        self.fc_out_linear = nn.Linear(self.h * self.d_v, self.D)

    def forward(self, batch_size, values, keys, query, mask):

        values = self.linear_values(values).view(batch_size, -1, self.h, self.d_v)
        values = values.permute(0, 2, 1, 3)  # values = values reshape/transpose to (b, h, n, d_v)
        keys = self.linear_keys(keys).view(batch_size, -1, self.h, self.d_k)
        keys = keys.permute(0, 2, 1, 3)
        queries = self.linear_queries(query).view(batch_size, -1, self.h, self.d_k)
        queries = queries.permute(0, 2, 1, 3)  # queries = query reshape/transpose to (b, h, n, d_k)

        # Einsum does batch matrix multiplication for query*keys for each training example
        # with every other training example --> similarity between Q and K
        keys_T = keys.permute(0, 1, 3, 2)  # keys_T = keys transpose to (b, h, d_k, n)
        scores = torch.einsum("bhnk,bhkm->bhnm", [queries, keys_T])

        if mask is not None:
            mask = mask.unsqueeze (1)  # (b, 1, n, n)
            expanded_mask = mask.expand (batch_size, self.h, -1, -1)  # (b, h, n, n)
            scores = scores.masked_fill_(expanded_mask, float("-1e20"))

        attention = torch.softmax(scores, dim = -1)  # attention = shape (b, h, n, n)

        multi_attention = torch.einsum("bhnn,bhnv->bhnv", [attention, values])
        multi_out = multi_attention.permute(0, 2, 1, 3).contiguous()  # out = out reshape/transpose to (b, n, h, d_v)
        multi_out = multi_out.view(batch_size, -1, self.h * self.d_v)  # out = out reshape to (b, n, h * d_v)

        out = self.fc_out_linear(multi_out) # out = out linear to (b, n, d_model)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, expansion_size):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.expansion_size = expansion_size
        self.linear = nn.Linear(self.embed_size, self.expansion_size)

    def forward(self, x):
        self.linear.to(x.dtype)
        out = self.linear(x)
        return out

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dim_k, dim_v, dropout, device, forward_expansion = 32):
        super(TransformerDecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads, dim_k, dim_v)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def forward(self, batch_size, x, last_out, future_mask):

        attention = self.attention(batch_size, last_out, last_out, last_out, future_mask)
        query = self.dropout(self.norm1(attention + x))
        attention = self.attention(batch_size, x, x, query, None)
        attention_out = self.dropout(self.norm2(self.feed_forward(attention) + attention))
        out = self.norm3(attention_out + self.feed_forward(attention_out))
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, encoding_size, heads, dim_k, dim_v, sequence_length, predict_length, num_layers, dropout, device):
        super(TransformerDecoder, self).__init__()
        self.to(device)
        self.sequence_length = sequence_length
        self.predict_length = predict_length
        self.embed_size = embed_size
        self.encoding_size = encoding_size
        self.encoding = TransformerEncoder(embed_size = self.embed_size + 1, expansion_size = self.encoding_size)
        self.positional_encoding = PositionalEncoding(D = self.embed_size, dropout = dropout, N = 144)
        self.additive_positional_encoding = Additive_PositionalEncoding(D = self.embed_size, N = 144)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    self.encoding_size,
                    heads,
                    dim_k,
                    dim_v,
                    dropout,
                    device
                )
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(sequence_length, predict_length)
        self.decoding = TransformerEncoder(embed_size = self.encoding_size, expansion_size = self.embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_size, x, future_mask, status = 'train'):
        if status == 'train' or status == 'predict':

            # when use traditional positional encoding only
            # x = self.positional_encoding(x)

            # when use additive positional encoding only
            x = self.additive_positional_encoding(x)

            # when use hybrid positional encoding
            # x = self.positional_encoding(x)
            # x = self.additive_positional_encoding(x)
            x = self.encoding(x)
            out = x

            for layer in self.layers:
                out = layer (batch_size, x, out, future_mask)  # (b, n, d)


            out = self.decoding(out) # （b, n, d）-> (b, n, D)

        return out




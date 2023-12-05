import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mask = mask.unsqueeze(1)  # (b, 1, 6, 6)
        expanded_mask = mask.expand(-1, 2, -1, -1) # (b, h, 6, 6)

        # Einsum does batch matrix multiplication for query*keys for each training example
        # with every other training example --> similarity between Q and K
        keys_T = keys.permute(0, 1, 3, 2)  # keys_T = keys transpose to (b, h, d_k, n)
        scores = torch.einsum("bhnk,bhkm->bhnm", [queries, keys_T])
        if mask is not None:
            attention = scores.masked_fill(expanded_mask == 0, float("-1e20"))

        attention = torch.softmax(attention, dim = -1)  # attention = shape (b, h, n, n)

        multi_attention = torch.einsum("bhnn,bhnv->bhnv", [attention, values])
        multi_out = multi_attention.permute(0, 2, 1, 3).contiguous()  # out = out reshape/transpose to (b, n, h, d_v)
        multi_out = multi_out.view(batch_size, -1, self.h * self.d_v)  # out = out reshape to (b, n, h * d_v)

        out = self.fc_out_linear(multi_out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dim_k, dim_v, dropout, device, forward_expansion = 32):
        super(TransformerDecoderLayer, self).__init__()
        self.device = device
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

    def forward(self, batch_size, x, last_out, src_mask, trg_mask):
        attention = self.attention(batch_size, last_out, last_out, last_out, trg_mask)
        query = self.dropout(self.norm1(attention + last_out))
        attention = self.attention(batch_size, query, x, x, src_mask)
        attention_out = self.dropout(self.norm2(self.feed_forward(attention) + attention))
        out = self.norm3(attention_out + self.feed_forward(attention_out))
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, heads, dim_k, dim_v, num_layers, dropout, device):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_size,
                    heads,
                    dim_k,
                    dim_v,
                    dropout,
                    device
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.initial_y = True

    def forward(self, batch_size, x, src_mask, trg_mask, y=None):
        if self.initial_y:
            out = x
            self.initial_y = False
        else:
            out = y

        for layer in self.layers:
            out = layer(batch_size, x, out, src_mask, trg_mask)

        return out




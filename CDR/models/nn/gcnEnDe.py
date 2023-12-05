import torch
import torch.nn as nn

from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj

# from ...utils.functional import recons_cross_loss
from ...utils import functional

class GCNEnDeBase(nn.Module):
    """
    Parameters
    ----------
        in_dim : int
        Input dimension of model.
    hid_dim :  int
       Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the encoder, the other half (ceil) of the layers are
       for decoders. Default: ``4``.
    dropout : float, optional
       Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
       Activation function if not None.
       Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    **kwargs : optional
        Additional arguments for the backbone.

    """

    def __init__(self,
                 in_dim,
                 hid_dim = 64,
                 num_layers = 4,
                 dropout = 0.,
                 act = torch.nn.functional.relu,
                 backbone = GCN,
                 **kwargs):
        super(GCNEnDeBase,self).__init__()

        self.encoder = backbone(in_channels = in_dim,
                                hidden_channels = hid_dim,
                                num_layers = num_layers,
                                out_channels = hid_dim,
                                dropout = dropout,
                                act = act,
                                **kwargs)

        self.decoder = backbone(in_channels = hid_dim,
                                hidden_channels = hid_dim,
                                num_layers = num_layers,
                                out_channels = in_dim,
                                dropout = dropout,
                                act = act,
                                **kwargs)

        self.loss_func = functional.recons_cross_loss
        self.emb = None

    def forward(self, x, edge_index):
        # encode feature matrix
        self.emb = self.encoder(x, edge_index)

        #reconstruct latent features
        x_ = self.decoder(self.emb, edge_index)

        return x_

    @staticmethod
    def process_graph(data):
        data.s = to_dense_adj(data.edge_index)[0]
